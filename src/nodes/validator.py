"""Orchestral Machine -- Validator Node.

QUARANTINE quality gate for archives and state transitions.  The Validator
ensures that only verified, high-quality data is stored in long-term memory
and accessible to the Architect.

Responsibilities:
    - Accuracy check of Archivist output (approve or reject)
    - Control flow signalling (``ready_for_next``, ``approved_hard_reset``,
      ``unapproved_archived_summary``, ``unapproved_meta_summary``)
    - Hard Reset emission when ``entry_status`` indicates confirmed failure
      AND both archived refs are present

The Validator is a ``decision_authority`` role that CAN set ``approved_*``
keys.  It acts as a SIGNAL emitter -- it MUST NOT clear variables, delete
files, or reset counters.  The Graph Controller executes all mutations
atomically.
"""

import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    MODEL_MAPPING,
    NODE_REGEN_MAX,
    NODE_TIMEOUT_DEFAULT,
)
from src.enforcement import enforce_constitutional_rules
from src.llm_factory import build_chat_llm, invoke_node_llm
from src.nodes.base import BaseNode
from src.schemas import ValidatorOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template -- injected with live schema at runtime
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
SYSTEM: You are the VALIDATOR role in the Orchestral Machine - a Validation and State Control Agent. You are the Quality Assurance gate for Memory (Archives). Your sole purpose is to ensure that only verified, high-quality data is stored in long-term memory and accessible to the Architect.

CONTEXT:
- You receive state['archived_summary_ref'] and state['archived_meta_summary_ref'].
- You control state transitions and Hard Reset execution.

YOUR_MANDATE:
1. **Accuracy Check:** Read the Archivist's output'. Approve or if there are hallucinations or factual distortions, REJECT it. Validate that archived summaries are factually accurate and complete.
2. **Control Flow:**
    - If approved: Set status to 'ready_for_next'. The graph will resume using `state.entry_status`.
	- If You find issues with archived summaries (looks incomplete or hallucinated=rejected), return a JSON with the appropriate status ("unapproved_archived_summary" or "unapproved_meta_summary") = downgrade back to Archivist, include a `reason` string explaining the rejection, and a `correction_instructions[]` array with specific, actionable instructions to guide the Archivist retry.
    - If `archived_meta_summary_ref` is present, VALIDATE IT. If it is missing, VALIDATE `archived_summary_ref` (A1). Do not confuse the two."
	- If Hard Reset is required (based on failure depth): emit Validator Result.status == "approved_hard_reset". The Validator MUST NOT perform state mutations. The Graph Controller is the component that will execute the Hard Reset mutation sequence atomically upon receiving this decision.

HARD_RESET_CONDITIONS:
- status == confirmed_needs_rewrite AND both archived refs present
- status == confirmed_L3 AND both archived refs present.

  **Authority Limit:**
 - You act as a SIGNAL emitter. You MAY ask "hard_reset" by setting status: "approved_hard_reset". You DO NOT clear variables, delete files, or reset counters.  The Graph Controller will detect your status and execute the Hard Reset mutation sequence (clearing plan/code, resetting counters) atomically.

STRICT_PROHIBITIONS:
- You MUST NOT modify archived content.
- You MUST NOT select next nodes outside defined transitions.
- You MUST NOT change state['task'].
- You MUST NOT reset counters or variables yourself.
- You MUST NOT perform retries or mutate archival artifacts; the Graph Controller will route the retry to the appropriate Archivist node.

OUTPUT_FORMAT:
Return a JSON object matching the VALIDATOR schema. Strict JSON only.
{schema}

- AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the global state yourself. The Graph Controller will append this record."""


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


@enforce_constitutional_rules
def validator_node(state: dict) -> dict:
    """Validator node -- QUARANTINE quality gate for archives and transitions.

    Validates Archivist output for factual accuracy and completeness.
    Signals state transitions (ready_for_next, approved_hard_reset) and
    emits rejection feedback for Archivist retries.

    Args:
        state: Full graph state dictionary.

    Returns:
        Flat dict with ``approved_*`` flags, ``status``, and audit
        metadata for the Graph Controller to merge into state.
    """
    archived_summary_ref = state.get("archived_summary_ref")
    archived_meta_summary_ref = state.get("archived_meta_summary_ref")
    entry_status: str | None = state.get("entry_status")
    task_id: str = state["task_id"]

    # ------------------------------------------------------------------
    # 1. Initialize BaseNode and generating seed/prompt
    # ------------------------------------------------------------------
    node = BaseNode(ValidatorOutput, _SYSTEM_PROMPT_TEMPLATE)
    seed = node.generate_seed(task_id)
    system_prompt = node.build_system_prompt()

    # ------------------------------------------------------------------
    # 4. Build human message -- include ALL readable fields
    # ------------------------------------------------------------------
    human_parts: list[str] = []

    if archived_summary_ref is not None:
        human_parts.append(
            f"Archived Summary Ref (A1 output):\n"
            f"{json.dumps(archived_summary_ref, indent=2) if isinstance(archived_summary_ref, dict) else str(archived_summary_ref)}"
        )
    else:
        human_parts.append("Archived Summary Ref: None")

    if archived_meta_summary_ref is not None:
        human_parts.append(
            f"\nArchived Meta-Summary Ref (A2 output):\n"
            f"{json.dumps(archived_meta_summary_ref, indent=2) if isinstance(archived_meta_summary_ref, dict) else str(archived_meta_summary_ref)}"
        )
    else:
        human_parts.append("\nArchived Meta-Summary Ref: None")

    human_parts.append(
        f"\nEntry Status (captured before Validator): {entry_status or 'None'}"
    )
    human_parts.append(f"\nTask ID: {task_id}")
    human_parts.append(f"Seed: {seed}")

    human_content = "\n\n".join(human_parts)

    # ------------------------------------------------------------------
    # 5. Create LLM instance (NODE_TIMEOUT_DEFAULT -- not in LONG_TIMEOUT_ROLES)
    # ------------------------------------------------------------------
    model_id = MODEL_MAPPING["VALIDATOR"]
    llm = build_chat_llm(
        model_id=model_id,
        timeout=NODE_TIMEOUT_DEFAULT,
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
        output_schema_cls=ValidatorOutput,
        seed=seed,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 7. Handle total failure
    # ------------------------------------------------------------------
    if parsed_output is None:
        logger.error("Validator node exhausted all attempts.")
        now = datetime.utcnow().isoformat()
        return {
            "status": "FAIL",
            "error_logs": state.get("error_logs", [])
            + [
                {
                    "node": "VALIDATOR",
                    "error": "Failed to generate valid ValidatorOutput after retries.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "validator_failed",
                    "detail": "Failed to generate valid ValidatorOutput after retries.",
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

    # Build approved_* flags based on status
    approved_summary: bool = parsed_output.status in (
        "ready_for_next",
        "approved_hard_reset",
        "COMPLETED",
    )
    approved_meta: bool = parsed_output.status in (
        "ready_for_next",
        "approved_hard_reset",
        "COMPLETED",
    )
    approved_hard_reset: bool = parsed_output.status == "approved_hard_reset"

    # Build result dict
    result: dict = {
        "approved_summary": approved_summary,
        "approved_meta": approved_meta,
        "approved_hard_reset": approved_hard_reset,
        "status": parsed_output.status,
        "notes": parsed_output.notes,
        "audit": parsed_output.audit.model_dump() if parsed_output.audit else {},
        "error_logs": state.get("error_logs", []),
    }

    # For rejection statuses: expose correction_instructions for Graph Controller
    # to extract and set as archivist_feedback (List[str]) per State Mutation Principle.
    # Validator does NOT write archivist_feedback directly.
    if parsed_output.status in (
        "unapproved_archived_summary",
        "unapproved_meta_summary",
    ):
        result["correction_instructions"] = parsed_output.correction_instructions
        result["rejection_reason"] = parsed_output.reason

    # Determine event description based on status
    if parsed_output.status == "ready_for_next":
        event_name = "validator_approved"
        event_detail = (
            f"Validator approved archives, ready for next "
            f"(entry_status={entry_status}): "
            f"{parsed_output.reason[:100]}"
        )
    elif parsed_output.status == "approved_hard_reset":
        event_name = "validator_approved_hard_reset"
        event_detail = (
            f"Validator approved Hard Reset "
            f"(entry_status={entry_status}): "
            f"{parsed_output.reason[:100]}"
        )
    elif parsed_output.status == "HIR":
        event_name = "validator_hir"
        event_detail = f"Validator escalated to HIR: {parsed_output.reason[:120]}"
    elif parsed_output.status == "COMPLETED":
        event_name = "validator_completed"
        event_detail = f"Validator signalled COMPLETED: {parsed_output.reason[:120]}"
    elif parsed_output.status == "unapproved_archived_summary":
        event_name = "validator_rejected_a1"
        event_detail = f"Validator rejected A1 summary: {parsed_output.reason[:120]}"
    elif parsed_output.status == "unapproved_meta_summary":
        event_name = "validator_rejected_a2"
        event_detail = (
            f"Validator rejected A2 meta-summary: {parsed_output.reason[:120]}"
        )
    else:
        event_name = "validator_decision"
        event_detail = (
            f"Validator status={parsed_output.status}: {parsed_output.reason[:120]}"
        )

    result["event_log"] = state.get("event_log", []) + [
        {
            "event": event_name,
            "detail": event_detail,
            "timestamp": now,
        }
    ]

    result["active_model"] = model_id
    result["updated_at"] = now

    return result
