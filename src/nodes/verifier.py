"""Orchestral Machine -- Verifier Node.

Meta-Control Layer for critical decisions. The Verifier confirms or rejects
``NEEDS_REWRITE`` and ``error_L3`` claims to prevent infinite loops and
maintain constitutional integrity.  It also performs schema validation and
file completeness checks against the Architect plan.

The Verifier does NOT generate code, plans, or fixes.  It classifies
existing failures as either execution failures (local L1/L2) or
architectural failures (L3) and recommends the appropriate recovery path.

The Verifier is a ``decision_authority`` role whose exclusive key is
``verifier_feedback``.
"""

import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    MAX_CORRECTION_ATTEMPTS,
    MODEL_MAPPING,
    NODE_REGEN_MAX,
    NODE_TIMEOUT_DEFAULT,
)
from src.enforcement import enforce_constitutional_rules
from src.llm_factory import build_chat_llm, invoke_node_llm
from src.nodes.base import BaseNode
from src.schemas import VerifierOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template -- injected with live schema at runtime
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
SYSTEM: You are the VERIFIER role in the Orchestral Machine — Meta-Control Layer for critical decisions. You MUST NOT generate code, plans, or fixes. Your role is to confirm or reject NEEDS_REWRITE and L3_ARCHITECTURE claims to prevent infinite loops and maintain constitutional integrity and to perform schema/file completeness checks.

CONTEXT:
- You are invoked when critical decisions require confirmation.
- You receive:
	- state['plan']
	- state['code']
	- state['error_logs'] (list)
	- The specific claim under review (error_L3 or NEEDS_REWRITE).

YOUR_MANDATE:
1. **Schema Check:** JSON schema validation of plan, code structure and reviewer/tester outputs.
   - Check for constitutional violations (role boundary breaches).
   - Ensure all outputs are complete and contain no placeholders or incomplete artifacts.
   - Verify that all files declared in plan exist and are non-empty.
2. **L3 Verification:** If Reviewer claims 'error_L3', you must confirm or reject this based on evidence. Rejection prevents unnecessary Hard Resets.
   - Evaluate whether the Reviewer's L3 claim is supported by evidence in error_logs and code (produce conservative binary decision).
3. **Rewrite Necessity:** If Corrector claims 'NEEDS_REWRITE', confirm status='confirmed_needs_rewrite' if the architectural debt is too high for local fixes, or reject as event='execution_failure' if Hard Reset is unnecessary and Corrector has to start over.

DECISION_CRITERIA:
- L3 is CONFIRMED only if the issue genuinely cannot be fixed without plan changes.
- NEEDS_REWRITE is CONFIRMED only if correction_attempt > {max_correction} AND issues persist.
- Downgrade to L1/L2 if local fixes are possible.

 **DECISION LOGIC:**
- Execution Failure vs Architectural Failure:
  - Execution Failure: If failures are confined to failing tests, reproducible runtime exceptions, or localized L1/L2 issues where code matches plan and changes are surgical, return "execution_failure".
  - Architectural Failure (L3): If failures require changes to file structure, public interfaces, major design decisions, or plan constraints, return "confirmed_L3".
- correction_attempt reset logic:
  - If you return "execution_failure", Graph Controller will reset state['correction_attempt'] = 0 and route to CORRECTOR_C1 to attempt a new correction cycle; You should include "reset_recommendation": true in your JSON.
  - If you return "confirmed_needs_rewrite" or "confirmed_L3", you must not perform the reset; instead emit the decision and let the Graph Controller trigger archival and hard_reset workflows.
- Evidence requirement:
  - For each decision, include array evidence[] with pointers to tests, failing traces, and reviewer issues justifying the classification.

STRICT_PROHIBITIONS:
- You MUST NOT generate code, plans, or fixes.
- You MUST NOT act as a quality gate for normal code review.
- You MUST NOT propose architectural alternatives.

OUTPUT_FORMAT:
Return a JSON object matching the VERIFIER schema with clear reasoning. Strict JSON only.
{schema}

If you return confirmed_needs_rewrite or confirmed_L3, set no mutable state yourself — instead, emit that decision for graph controller to archive and route (Archivist → Validator → Architect).

- AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the global state yourself. The Graph Controller will append this record."""


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


@enforce_constitutional_rules
def verifier_node(state: dict) -> dict:
    """Verifier node -- Meta-Control Layer for critical decisions.

    Confirms or rejects NEEDS_REWRITE and L3_ARCHITECTURE claims.
    Performs schema validation and file completeness checks.

    Args:
        state: Full graph state dictionary.

    Returns:
        Flat dict with ``verifier_feedback``, ``status``, and audit
        metadata for the Graph Controller to merge into state.
    """
    plan: dict = state.get("plan", {}) or {}
    code: dict[str, str] = state["code"]
    error_logs: list = state.get("error_logs", [])
    correction_attempt: int = state.get("correction_attempt", 0)
    task_id: str = state["task_id"]

    # ------------------------------------------------------------------
    # 0. [v0.6.9] Require sandbox
    # ------------------------------------------------------------------
    from src.sandbox import get_sandbox

    sandbox = get_sandbox(task_id)
    if sandbox is None:
        now = datetime.utcnow().isoformat()
        return {
            "verifier_feedback": None,
            "status": "HIR",
            "notes": "",
            "error_logs": error_logs
            + [
                {
                    "node": "VERIFIER",
                    "error": "Docker sandbox unavailable.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "verifier_failed",
                    "detail": "No sandbox registered for task; HIR triggered.",
                    "timestamp": now,
                }
            ],
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 1. Initialize BaseNode and generating seed/prompt
    # ------------------------------------------------------------------
    node = BaseNode(VerifierOutput, _SYSTEM_PROMPT_TEMPLATE)
    seed = node.generate_seed(task_id)
    system_prompt = node.build_system_prompt(
        max_correction=MAX_CORRECTION_ATTEMPTS,
    )

    # ------------------------------------------------------------------
    # 2. Build human message -- include ALL readable fields
    # ------------------------------------------------------------------
    human_parts: list[str] = [
        f"Plan (Architect's specification):\n{json.dumps(plan, indent=2)}",
        "\nCode (files mapping):",
    ]
    for filename, content in code.items():
        human_parts.append(f"--- {filename} ---\n{content}")

    human_parts.append(f"\nError Logs:\n{json.dumps(error_logs, indent=2)}")
    human_parts.append(f"\nCorrection Attempt: {correction_attempt}")
    human_parts.append(f"Task ID: {task_id}")
    human_parts.append(f"Seed: {seed}")

    human_content = "\n\n".join(human_parts)

    # ------------------------------------------------------------------
    # 3. Create LLM instance (NODE_TIMEOUT_DEFAULT -- not in LONG_TIMEOUT_ROLES)
    # ------------------------------------------------------------------
    model_id = MODEL_MAPPING["VERIFIER"]
    llm = build_chat_llm(
        model_id=model_id,
        timeout=NODE_TIMEOUT_DEFAULT,
        seed=int(seed, 16) % (2**31),
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    parsed_output = invoke_node_llm(
        llm=llm,
        model_id=model_id,
        messages=messages,
        output_schema_cls=VerifierOutput,
        seed=seed,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 4. Handle total failure
    # ------------------------------------------------------------------
    if parsed_output is None:
        logger.error("Verifier node exhausted all attempts.")
        now = datetime.utcnow().isoformat()
        return {
            "verifier_feedback": None,
            "status": "FAIL",
            "error_logs": error_logs
            + [
                {
                    "node": "VERIFIER",
                    "error": "Failed to generate valid VerifierOutput after retries.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "verifier_failed",
                    "detail": "Failed to generate valid VerifierOutput after retries.",
                    "timestamp": now,
                }
            ],
            "active_model": model_id,
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 5. Build successful state update
    # ------------------------------------------------------------------
    now = datetime.utcnow().isoformat()

    # Determine event description based on status
    if parsed_output.status == "confirmed_L3":
        event_name = "verifier_confirmed_L3"
        event_detail = (
            f"Verifier confirmed L3 architectural failure: {parsed_output.reason[:120]}"
        )
    elif parsed_output.status == "disagree_L3":
        event_name = "verifier_disagree_l3"
        event_detail = (
            f"Verifier downgraded L3 claim to L1/L2: {parsed_output.reason[:120]}"
        )
    elif parsed_output.status == "confirmed_needs_rewrite":
        event_name = "verifier_confirmed_needs_rewrite"
        event_detail = (
            f"Verifier confirmed NEEDS_REWRITE at attempt "
            f"{correction_attempt}: {parsed_output.reason[:100]}"
        )
    elif parsed_output.status == "execution_failure":
        event_name = "verifier_execution_failure"
        event_detail = (
            f"Verifier classified as execution failure "
            f"(reset_recommendation={parsed_output.reset_recommendation}): "
            f"{parsed_output.reason[:100]}"
        )
    else:
        event_name = "verifier_decision"
        event_detail = (
            f"Verifier status={parsed_output.status}: {parsed_output.reason[:120]}"
        )

    return {
        "verifier_feedback": parsed_output.model_dump(),
        "status": parsed_output.status,
        "notes": getattr(parsed_output, "notes", "") or "",
        "error_logs": error_logs,
        "event_log": state.get("event_log", [])
        + [
            {
                "event": event_name,
                "detail": event_detail,
                "timestamp": now,
            }
        ],
        "active_model": model_id,
        "updated_at": now,
    }
