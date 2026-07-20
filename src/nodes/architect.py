"""Orchestral Machine — Architect Node.

The first node in the LangGraph state machine. Analyzes the user's task
and produces a structured, versioned implementation plan in strict JSON
(Pydantic-valid) format. The plan becomes the single source of truth for
all downstream nodes (CODER, REVIEWER, TESTER, etc.).

The Architect MUST NOT write code, simplify requirements, or mutate any
state key outside its exclusive scope (``plan``, ``execution_strategy``).
"""

import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    MODEL_MAPPING,
    NODE_REGEN_MAX,
    NODE_TIMEOUT_LONG,
)
from src.enforcement import enforce_constitutional_rules
from src.llm_factory import build_chat_llm, invoke_node_llm
from src.nodes.base import BaseNode
from src.schemas import ArchitectOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template — injected with live schema at runtime
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
SYSTEM:
- You are the ARCHITECT role in the Orchestral Machine. Act as a Senior Software Architect specializing in distributed systems design, who responsible for high-level planning and structural integrity.
- Your sole authority is to analyze the immutable state['task'] and produce a complete, structured, versioned implementation plan in strict JSON (Pydantic-valid) format under state['plan']. You MUST NOT write any code, modify state except to output the plan, or simplify/omit requirements from state['task'].

CONTEXT:
- **Read-Only Task:** You receive a task definition in state['task'] which is IMMUTABLE and ABSOLUTE (read-only, original user request).
- **Fail-Safe Strategy:**  You may receive state['archived_meta_summary_ref'] containing validated failure history from previous attempts.
- If state['code'] is non-empty and the task describes modifications to an existing product, treat state['code'] as an additional read-only input: analyze code to identify exact files/locations impacted and produce a plan that enumerates targeted changes (filename, function/class, line ranges or anchors).
- You must explicitly include in the Plan a "modification_map" mapping each required change to a file and location specifier, and must indicate "delta" vs "rewrite" for each deliverable.

YOUR_MANDATE:
1. Analyze state['task'] completely and produce a structured implementation plan.
   - Respect Immutable Requirements (state['task'] read-only).
2. If archived_meta_summary_ref exists, study it to understand previous architectural failures and explicitly avoid repeating those failure modes.
3. Your Plan will be consumed directly by the CODER role in the next graph step. Therefore the Plan must be immediately actionable: include file structure, explicit file mappings, expected file contents signatures, dependencies, logical steps, complete, actionable, deterministic inputs (seeds), and any test targets. The Plan is the single source of truth for Coder's implementation.
   - Use only factual, actionable language; avoid speculative content.

STRICT_PROHIBITIONS:
- You MUST NOT propose or write any source code, placeholders, or partial solutions; provide only a full plan document.
- You MUST NOT simplify, reinterpret, or weaken requirements from state['task'].
- You MUST NOT propose "easier alternatives" to achieve test success.
- You MUST NOT change architecture or public interfaces beyond the plan metadata.

OUTPUT_FORMAT:
Return a JSON object matching the ARCHITECT schema. Strict JSON only.
{schema}

- Produce a Plan JSON exactly matching the Plan contract (task_id, objective, constraints, deliverables, steps[], version, metadata).
- Clear file structure with exact filenames.
- Step-by-step implementation instructions. Steps must be numbered, finite (<=30), and each step must include a deliverable file mapping (filename → purpose).
- All constraints and edge cases from state['task']
- Include deterministic decisions and explicit rationale for tradeoffs (concise), and reference any archived meta-summary entries that influenced choices.
- Ensure plan enforces determinism constraints (temperature=0.0, seed derivation from task_id) and lists required test targets.
- AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the global state yourself. The Graph Controller will append this record.

If previous attempts failed, your plan MUST explicitly state how it differs from failed approaches (validated_archive) and how this plan avoids them.

Produce only the JSON plan object as your final output (no surrounding commentary)."""


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


@enforce_constitutional_rules
def architect_node(state: dict) -> dict:
    """Architect node — analyzes the task and produces a structured plan.

    Args:
        state: Full graph state dictionary.

    Returns:
        Flat dict with ``plan``, ``status``, and audit metadata for the
        Graph Controller to merge into state.
    """
    task: str = state["task"]
    task_id: str = state["task_id"]

    # ------------------------------------------------------------------
    # 1. Initialize BaseNode and generating seed/prompt
    # ------------------------------------------------------------------
    node = BaseNode(ArchitectOutput, _SYSTEM_PROMPT_TEMPLATE)
    seed = node.generate_seed(task_id)
    system_prompt = node.build_system_prompt()

    # ------------------------------------------------------------------
    # 4. Build human message from state inputs
    # ------------------------------------------------------------------
    human_parts: list[str] = [f"Task: {task}", f"Task ID: {task_id}", f"Seed: {seed}"]

    # Optional: validated failure history
    archived_meta = state.get("archived_meta_summary_ref")
    if archived_meta:
        human_parts.append(
            f"Archived Meta-Summary (validated failure history):\n"
            f"{json.dumps(archived_meta, indent=2) if isinstance(archived_meta, dict) else str(archived_meta)}"
        )

    # Optional: previous archival reference
    archived_summary = state.get("archived_summary_ref")
    if archived_summary:
        human_parts.append(
            f"Archived Summary Reference:\n"
            f"{json.dumps(archived_summary, indent=2) if isinstance(archived_summary, dict) else str(archived_summary)}"
        )

    # Optional: existing code for modification tasks
    code = state.get("code")
    if code:
        human_parts.append("Existing Code (read-only, for modification planning):")
        for filename, content in code.items():
            human_parts.append(f"--- {filename} ---\n{content}")

    # Loop iteration context
    loop_iteration = state.get("loop_iteration", 0)
    if loop_iteration > 0:
        human_parts.append(
            f"Loop Iteration: {loop_iteration} (this is a re-plan after previous failure)"
        )

    human_content = "\n\n".join(human_parts)

    # ------------------------------------------------------------------
    # 5. Create LLM instance
    # ------------------------------------------------------------------
    model_id = MODEL_MAPPING["ARCHITECT"]
    llm = build_chat_llm(
        model_id=model_id,
        timeout=NODE_TIMEOUT_LONG,
        seed=int(seed, 16) % (2**31),
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    # ------------------------------------------------------------------
    # 6. Invoke LLM with JSON parse retry logic
    # ------------------------------------------------------------------
    parsed_output = invoke_node_llm(
        llm=llm,
        model_id=model_id,
        messages=messages,
        output_schema_cls=ArchitectOutput,
        seed=seed,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 7. Handle total failure
    # ------------------------------------------------------------------
    if parsed_output is None:
        logger.error("Architect node exhausted all attempts.")
        now = datetime.utcnow().isoformat()
        return {
            "plan": None,
            "status": "FAIL",
            "error_logs": state.get("error_logs", [])
            + [
                {
                    "node": "ARCHITECT",
                    "error": "Failed to generate valid ArchitectOutput after retries.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "architect_failed",
                    "detail": "Failed to generate valid ArchitectOutput after retries.",
                    "timestamp": now,
                }
            ],
            "active_model": model_id,
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 8. Build successful state update
    # ------------------------------------------------------------------
    now = datetime.utcnow().isoformat()

    return {
        "plan": parsed_output.model_dump(),
        "status": parsed_output.status,
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "architect_plan_ready",
                "detail": f"Plan v{parsed_output.version} with {len(parsed_output.steps)} steps",
                "timestamp": now,
            }
        ],
        "active_model": model_id,
        "updated_at": now,
    }
