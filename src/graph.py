"""Orchestral Machine — Graph Controller.

The central orchestration engine that routes between nodes, enforces rules,
manages counters, and performs all state mutations atomically.

The Graph Controller is the SOLE authority for:
    - Routing between nodes based on validated outputs and counters.
    - Enforcing orchestration rules, loop limits, and retry policies.
    - Performing all global state mutations, including Hard Reset.
    - Enforcing role boundaries and runtime permissions via Data Access Matrix.
    - Mapping AuditMetadata (from nodes) → AuditEntry (in state).

Exports:
    graph  — StateGraph definition (uncompiled)
    app    — compiled LangGraph runnable
"""

import contextvars
import logging
import threading
from typing import Any, Callable, Dict, List

from langgraph.graph import END, StateGraph

from src.access_control import (
    _filter_state_for_node,
    _strip_controller_exclusive_fields,
)
from src.config import (
    LONG_TIMEOUT_ROLES,
    MAX_ERROR_LOGS_IN_STATE,
    MAX_INFRASTRUCTURE_RETRIES,
    MODEL_MAPPING,
    NODE_HARD_TIMEOUT_BUFFER,
    NODE_TIMEOUT_DEFAULT,
    NODE_TIMEOUT_LONG,
    RECURSION_LIMIT,
)
from src.enforcement import ConstitutionalViolation, validate_state_mutation
from src.graph_utils import (
    _ensure_dict,
    _make_audit_entry,
    _make_system_audit_entry,
    _now,
    _seed,
)
from src.nodes.architect import architect_node
from src.nodes.archivist_a1 import archivist_a1_node
from src.nodes.archivist_a2 import archivist_a2_node
from src.nodes.coder import coder_node
from src.nodes.corrector import corrector_node
from src.nodes.reviewer import reviewer_node
from src.nodes.tester import tester_node
from src.nodes.validator import validator_node
from src.nodes.verifier import verifier_node
from src.routing import (
    a2_feedback_to_a1_node,
    route_after_a2_feedback,
    route_after_architect,
    route_after_archivist_a1,
    route_after_archivist_a2,
    route_after_archivist_queue_assembly,
    route_after_coder,
    route_after_corrector,
    route_after_entry_status_capture,
    route_after_hard_reset,
    route_after_reviewer,
    route_after_tester,
    route_after_validator,
    route_after_validator_feedback_a1,
    route_after_validator_feedback_a2,
    route_after_verifier,
    route_after_verifier_reset,
    validator_feedback_to_a1_node,
    validator_feedback_to_a2_node,
    verifier_reset_node,
)
from src.state import GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph Controller Internal Metadata (SPEC-2)
# ---------------------------------------------------------------------------
_controller_metadata: Dict[str, str] = {
    "last_completed_node": "",
    "next_scheduled_node": "ARCHITECT",
    "system_state": "RUNNING",
}


def _rotate_error_logs(error_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the most recent MAX_ERROR_LOGS_IN_STATE entries."""
    if len(error_logs) > MAX_ERROR_LOGS_IN_STATE:
        return error_logs[-MAX_ERROR_LOGS_IN_STATE:]
    return error_logs


def _append_event(state: dict, event: str, detail: str) -> List[Dict[str, Any]]:
    """Create updated event_log with a new entry appended."""
    return state.get("event_log", []) + [
        {"event": event, "detail": detail, "timestamp": _now()},
    ]


def _extract_nested_field(output: dict, field: str):
    """Extract a field from node output, searching nested dicts if needed."""
    if field in output:
        return output[field]
    for val in output.values():
        if isinstance(val, dict) and field in val:
            return val[field]
    return None


# ---------------------------------------------------------------------------
# Model Fallback Wrapper (SPEC-13)
# ---------------------------------------------------------------------------


def _try_with_fallback(
    node_func: Callable,
    state: dict,
    node_name: str,
) -> dict:
    """Invoke a node with primary model, fallback to RESERVIST on infra error.

    On infrastructure failure (ConnectionError, HTTP 5xx simulated via
    generic Exception with relevant message), retries with RESERVIST model.
    If RESERVIST also fails, returns HIR state.
    """
    for attempt in range(MAX_INFRASTRUCTURE_RETRIES + 1):
        try:
            result = node_func(state)
            return result
        except (ConnectionError, OSError, TimeoutError) as exc:
            if attempt == 0:
                # First failure: try RESERVIST
                logger.warning(
                    "Infrastructure error in %s (attempt %d): %s. "
                    "Falling back to RESERVIST model.",
                    node_name,
                    attempt + 1,
                    exc,
                )
                state = dict(state)
                state["active_model"] = MODEL_MAPPING.get("RESERVIST", "")
                state.setdefault("system_flags", {})["fallback_active"] = True
            else:
                logger.error(
                    "Infrastructure error in %s (attempt %d): %s. Exhausted retries.",
                    node_name,
                    attempt + 1,
                    exc,
                )
        except Exception as exc:
            # Catch-all for unexpected errors during node execution
            logger.error(
                "Unexpected error in %s: %s",
                node_name,
                exc,
            )
            break

    # All retries exhausted → HIR
    logger.critical(
        "Model fallback exhausted for %s. Transitioning to HIR.",
        node_name,
    )
    return {
        "status": "HIR",
        "notes": "",
        "event_log": _append_event(
            state,
            "model_fallback_exhausted",
            f"Node {node_name}: primary and RESERVIST models both failed",
        ),
        "updated_at": _now(),
    }


# ---------------------------------------------------------------------------
# Node Wrapper (SPEC-3, SPEC-6, SPEC-17)
# ---------------------------------------------------------------------------


def _wrap_node(
    node_func: Callable,
    node_name: str,
) -> Callable:
    """Create a LangGraph-compatible wrapped node function.

    The wrapper:
    1. Filters state per Data Access Matrix.
    2. Calls the node with model fallback.
    3. Extracts AuditMetadata → AuditEntry.
    4. Strips controller-exclusive fields from node output.
    5. Applies counter logic.
    6. Rotates error_logs.
    7. Returns the clean state update with canonical audit_log entry.
    """

    def wrapped(state: dict) -> dict:
        state = _ensure_dict(state)

        # Resume skip: fast-forward to target node without executing intermediate nodes
        resume_target = state.get("resume_target")
        if resume_target and node_name != resume_target:
            logger.info("Resume: skipping %s (target=%s)", node_name, resume_target)
            return {}

        if resume_target and node_name == resume_target:
            logger.info("Resume: reached target node %s, executing normally", node_name)

        now = _now()
        task_id = state.get("task_id", "unknown")
        seed = _seed(task_id)

        # Update controller metadata
        _controller_metadata["next_scheduled_node"] = node_name

        # Clear notes for new node start (to avoid stale info in UI/Bot)
        state["notes"] = None

        # 1. Filter state per Data Access Matrix
        filtered_state = _filter_state_for_node(state, node_name)

        # 2. Call node with model fallback
        # --- Hard timeout safety net ---
        role = node_name.upper()
        if role in LONG_TIMEOUT_ROLES:
            hard_timeout = NODE_TIMEOUT_LONG + NODE_HARD_TIMEOUT_BUFFER
        else:
            hard_timeout = NODE_TIMEOUT_DEFAULT + NODE_HARD_TIMEOUT_BUFFER

        _timeout_result: Dict[str, Any] = {}
        _timeout_error: list = []
        _timeout_done = threading.Event()

        def _guarded_call():
            try:
                _timeout_result["output"] = _try_with_fallback(
                    node_func, filtered_state, node_name
                )
            except Exception as exc:
                _timeout_error.append(exc)
            finally:
                from src.llm_factory import get_llm_stats

                _timeout_result["llm_stats"] = get_llm_stats()
                _timeout_done.set()

        ctx = contextvars.copy_context()
        t = threading.Thread(target=ctx.run, args=(_guarded_call,), daemon=True)
        t.start()

        if not _timeout_done.wait(timeout=hard_timeout):
            logger.critical(
                "HARD TIMEOUT: Node %s exceeded %ds. Forcing HIR.",
                node_name,
                hard_timeout,
            )
            _controller_metadata["system_state"] = "HIR"
            return {
                "status": "HIR",
                "notes": "",
                "event_log": _append_event(
                    state,
                    "node_hard_timeout",
                    f"Node {node_name} exceeded hard timeout ({hard_timeout}s)",
                ),
                "updated_at": now,
            }

        if _timeout_error:
            raise _timeout_error[0]

        node_output = _timeout_result["output"]
        llm_stats = _timeout_result.get("llm_stats", {})
        # --- End hard timeout safety net ---

        # Check for immediate HIR from fallback
        if node_output.get("status") == "HIR" and "model_fallback_exhausted" in str(
            node_output.get("event_log", [])
        ):
            _controller_metadata["system_state"] = "HIR"
            return node_output

        # Extract nested audit/notes and promote for controller-level processing
        nested_audit = _extract_nested_field(node_output, "audit")
        if isinstance(nested_audit, dict):
            node_output["audit"] = nested_audit
        node_notes = _extract_nested_field(node_output, "notes") or ""

        # Override self-reported model_id with infrastructure-tracked actual model
        actual_model = llm_stats.get("actual_model", "")
        if actual_model:
            node_output.setdefault("audit", {})["model_id"] = actual_model

        # 3. Build canonical AuditEntry
        audit_entry = _make_audit_entry(node_name, filtered_state, node_output, seed)

        # 4. Strip controller-exclusive fields (including inline audit_log)
        cleaned_output = _strip_controller_exclusive_fields(node_output)

        # 5. Handle counter requests from nodes
        state_patch: Dict[str, Any] = {}

        # Enrich notes with LLM call stats for Telegram visibility
        stats_parts = []
        val_errors = llm_stats.get("validation_errors", 0)
        if val_errors > 0:
            total = llm_stats.get("total_attempts", val_errors)
            stats_parts.append(f"JSON: {val_errors}/{total} failed")
        if llm_stats.get("reservist_used"):
            primary = llm_stats.get("primary_model", "primary")
            short_primary = primary.split("/")[-1] if "/" in primary else primary
            stats_parts.append(f"{short_primary} failed -> RESERVIST")
        last_error = llm_stats.get("last_error", "")
        if last_error and llm_stats.get("fatal"):
            stats_parts.append(f"Fatal: {last_error[:80]}")
        if stats_parts:
            stats_suffix = " | ".join(stats_parts)
            if node_notes:
                cleaned_output["notes"] = f"{node_notes} | {stats_suffix}"
            else:
                cleaned_output["notes"] = stats_suffix
        elif node_notes:
            cleaned_output["notes"] = node_notes
        else:
            cleaned_output["notes"] = ""

        if llm_stats.get("reservist_used"):
            state_patch["event_log"] = _append_event(
                state,
                "reservist_activated",
                f"Node {node_name}: RESERVIST model used after primary failures",
            )

        # Corrector: request_increment_correction_attempt
        if node_output.get("request_increment_correction_attempt", False):
            new_correction = state.get("correction_attempt", 0) + 1
            state_patch["correction_attempt"] = new_correction
            logger.info(
                "Graph Controller: correction_attempt incremented to %d",
                new_correction,
            )

        # Archivist A1: increment_archival_attempt
        if node_output.get("increment_archival_attempt", False):
            new_archival = state.get("archival_attempt", 0) + 1
            state_patch["archival_attempt"] = new_archival
            logger.info(
                "Graph Controller: archival_attempt incremented to %d",
                new_archival,
            )

        # 6. Append Reviewer issues to error_logs if present
        if node_name == "REVIEWER" and node_output.get("status") in (
            "error_L1",
            "error_L2",
            "error_L3",
        ):
            issues = node_output.get("issues", [])
            if issues:
                current_logs = list(state.get("error_logs", []))
                for issue in issues:
                    if isinstance(issue, dict):
                        current_logs.append(issue)
                    else:
                        current_logs.append({"issue": str(issue), "source": "REVIEWER"})
                state_patch["error_logs"] = _rotate_error_logs(current_logs)

        # Append Tester failures to error_logs on FAIL
        if node_name == "TESTER" and node_output.get("status") == "FAIL":
            failures = node_output.get("failures", [])
            if failures:
                current_logs = list(state.get("error_logs", []))
                for failure in failures:
                    if isinstance(failure, dict):
                        current_logs.append(failure)
                    else:
                        current_logs.append(
                            {"failure": str(failure), "source": "TESTER"}
                        )
                state_patch["error_logs"] = _rotate_error_logs(current_logs)

        # 7. Build final state update
        # Merge cleaned node output with controller patches
        final_update: Dict[str, Any] = {}
        final_update.update(cleaned_output)
        final_update.update(state_patch)

        # Canonical audit_log entry (replaces any inline one)
        current_audit_log = list(state.get("audit_log", []))
        current_audit_log.append(audit_entry.model_dump())
        final_update["audit_log"] = current_audit_log

        # Ensure updated_at is set
        final_update["updated_at"] = now

        # Clear resume_target after executing the target node
        if resume_target and node_name == resume_target:
            final_update["resume_target"] = None

        # Update controller metadata
        _controller_metadata["last_completed_node"] = node_name

        logger.info(
            "Node %s completed with status=%s",
            node_name,
            node_output.get("status"),
        )

        return final_update

    wrapped.__name__ = f"wrapped_{node_name.lower()}"
    wrapped.__qualname__ = f"wrapped_{node_name.lower()}"
    return wrapped


# ---------------------------------------------------------------------------
# System Nodes (non-LLM, Graph Controller actions)
# ---------------------------------------------------------------------------


def hard_reset_node(state: dict) -> dict:
    """System node — executes Hard Reset mutation sequence (SPEC-14).

    Atomically:
    - Clears plan, code, feedback, error_logs, counters.
    - Increments loop_iteration.
    - Resets verifier_reset_used.
    - Preserves task, task_id, archived_summary_ref, archived_meta_summary_ref.
    - Appends audit_log entry.
    """
    state = _ensure_dict(state)
    now = _now()
    task_id = state.get("task_id", "unknown")
    new_loop = state.get("loop_iteration", 0) + 1

    logger.info(
        "HARD RESET executing. loop_iteration: %d → %d",
        state.get("loop_iteration", 0),
        new_loop,
    )

    _controller_metadata["last_completed_node"] = "HARD_RESET"
    _controller_metadata["next_scheduled_node"] = "ARCHITECT"

    return {
        "plan": None,
        "code": {},
        "correction_attempt": 0,
        "archival_attempt": 0,
        "meta_attempt": 0,
        "validation_attempt": 0,
        "loop_iteration": new_loop,
        "verifier_reset_used": False,
        "review_feedback": None,
        "test_results": None,
        "verifier_feedback": None,
        "applied_fixes": [],
        "error_logs": [],
        "archivist_feedback": None,
        "archivist_queue": None,
        "entry_status": None,
        "status": "rewrite_confirmed",
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "hard_reset_executed",
                "detail": f"Hard reset executed. loop_iteration now {new_loop}",
                "timestamp": now,
            }
        ],
        "audit_log": state.get("audit_log", [])
        + [
            _make_system_audit_entry(
                status="hard_reset_executed",
                prompt_summary=(
                    "Atomic hard reset: plan=None, code={}, "
                    "all counters reset, loop_iteration incremented"
                ),
                task_id=task_id,
            ),
        ],
        "updated_at": now,
    }


def entry_status_capture_node(state: dict) -> dict:
    """System node — captures current status into entry_status before Archival flow.

    The entry_status is a snapshot of the last non-archival routing status
    from Reviewer/Tester/Verifier that triggered archival flow.
    """
    state = _ensure_dict(state)
    current_status = state.get("status", "")
    now = _now()

    logger.info(
        "Capturing entry_status=%s before archival flow",
        current_status,
    )

    return {
        "entry_status": current_status,
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "entry_status_captured",
                "detail": f"entry_status set to '{current_status}' before archival",
                "timestamp": now,
            }
        ],
        "updated_at": now,
    }


def archivist_queue_assembly_node(state: dict) -> dict:
    """System node — assembles archivist_queue before ARCHIVIST_A1 invocation.

    Packages raw error_logs, session snapshot data, and relevant context
    for archival processing.
    """
    state = _ensure_dict(state)
    now = _now()

    queue: Dict[str, Any] = {
        "error_logs": list(state.get("error_logs", [])),
        "task_id": state.get("task_id", "unknown"),
        "task": state.get("task", ""),
        "status": state.get("status", ""),
        "entry_status": state.get("entry_status", ""),
        "correction_attempt": state.get("correction_attempt", 0),
        "loop_iteration": state.get("loop_iteration", 0),
        "code_snapshot": state.get("code", {}),
        "plan_snapshot": state.get("plan"),
        "assembled_at": now,
    }

    logger.info(
        "Archivist queue assembled with %d error_log entries", len(queue["error_logs"])
    )

    return {
        "archivist_queue": queue,
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "archivist_queue_assembled",
                "detail": f"Queue assembled with {len(queue['error_logs'])} error entries",
                "timestamp": now,
            }
        ],
        "updated_at": now,
    }


def hir_halt_node(state: dict) -> dict:
    """System node — transitions the graph to HIR / HALTED state."""
    state = _ensure_dict(state)
    now = _now()
    task_id = state.get("task_id", "unknown")

    _controller_metadata["system_state"] = "HIR"

    logger.critical("System transitioning to HIR. Halting execution.")

    from src.checkpoint import create_checkpoint

    try:
        create_checkpoint(
            graph_state=state,
            reason="HIR_AUTO",
            last_completed_node=_controller_metadata.get("last_completed_node", ""),
            next_scheduled_node=_controller_metadata.get("next_scheduled_node", ""),
            system_state="HIR",
            recursion_count=0,
            label="hir",
        )
    except Exception as _cp_exc:
        logger.warning("Failed to create HIR checkpoint: %s", _cp_exc)

    return {
        "status": "HIR",
        "system_flags": {
            **state.get("system_flags", {}),
            "halted": True,
        },
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "system_halted",
                "detail": "Human Intervention Required — system halted",
                "timestamp": now,
            }
        ],
        "audit_log": state.get("audit_log", [])
        + [
            _make_system_audit_entry(
                status="HIR",
                prompt_summary="System halted: Human Intervention Required",
                task_id=task_id,
            ),
        ],
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# Wrapped Node Functions
# ---------------------------------------------------------------------------

wrapped_architect = _wrap_node(architect_node, "ARCHITECT")
wrapped_coder = _wrap_node(coder_node, "CODER")
wrapped_reviewer = _wrap_node(reviewer_node, "REVIEWER")
wrapped_tester = _wrap_node(tester_node, "TESTER")
wrapped_corrector = _wrap_node(corrector_node, "CORRECTOR")
wrapped_verifier = _wrap_node(verifier_node, "VERIFIER")
wrapped_archivist_a1 = _wrap_node(archivist_a1_node, "ARCHIVIST_A1")
wrapped_archivist_a2 = _wrap_node(archivist_a2_node, "ARCHIVIST_A2")
wrapped_validator = _wrap_node(validator_node, "VALIDATOR")


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

graph = StateGraph(GraphState)

# --- Register LLM nodes (wrapped) ---
graph.add_node("ARCHITECT", wrapped_architect)
graph.add_node("CODER", wrapped_coder)
graph.add_node("REVIEWER", wrapped_reviewer)
graph.add_node("TESTER", wrapped_tester)
graph.add_node("CORRECTOR", wrapped_corrector)
graph.add_node("VERIFIER", wrapped_verifier)
graph.add_node("ARCHIVIST_A1", wrapped_archivist_a1)
graph.add_node("ARCHIVIST_A2", wrapped_archivist_a2)
graph.add_node("VALIDATOR", wrapped_validator)

# --- Register system nodes ---
graph.add_node("HARD_RESET", hard_reset_node)
graph.add_node("ENTRY_STATUS_CAPTURE", entry_status_capture_node)
graph.add_node("ARCHIVIST_QUEUE_ASSEMBLY", archivist_queue_assembly_node)
graph.add_node("HIR_HALT", hir_halt_node)
graph.add_node("VERIFIER_RESET_THEN_CORRECTOR", verifier_reset_node)
graph.add_node("A2_FEEDBACK_TO_A1", a2_feedback_to_a1_node)
graph.add_node("VALIDATOR_FEEDBACK_TO_A1", validator_feedback_to_a1_node)
graph.add_node("VALIDATOR_FEEDBACK_TO_A2", validator_feedback_to_a2_node)

# --- Entry point ---
graph.set_entry_point("ARCHITECT")

# --- Conditional edges ---
graph.add_conditional_edges(
    "ARCHITECT",
    route_after_architect,
    {
        "CODER": "CODER",
        "HIR_HALT": "HIR_HALT",
    },
)

graph.add_conditional_edges(
    "CODER",
    route_after_coder,
    {
        "REVIEWER": "REVIEWER",
        "HIR_HALT": "HIR_HALT",
    },
)

graph.add_conditional_edges(
    "REVIEWER",
    route_after_reviewer,
    {
        "TESTER": "TESTER",
        "CORRECTOR": "CORRECTOR",
        "VERIFIER": "VERIFIER",
        "HIR_HALT": "HIR_HALT",
    },
)

graph.add_conditional_edges(
    "TESTER",
    route_after_tester,
    {
        "ENTRY_STATUS_CAPTURE": "ENTRY_STATUS_CAPTURE",
        "CORRECTOR": "CORRECTOR",
        "HIR_HALT": "HIR_HALT",
    },
)

graph.add_conditional_edges(
    "CORRECTOR",
    route_after_corrector,
    {
        "REVIEWER": "REVIEWER",
        "VERIFIER": "VERIFIER",
        "HIR_HALT": "HIR_HALT",
    },
)

graph.add_conditional_edges(
    "VERIFIER",
    route_after_verifier,
    {
        "ENTRY_STATUS_CAPTURE": "ENTRY_STATUS_CAPTURE",
        "CORRECTOR": "CORRECTOR",
        "VERIFIER_RESET_THEN_CORRECTOR": "VERIFIER_RESET_THEN_CORRECTOR",
        "HIR_HALT": "HIR_HALT",
    },
)

graph.add_conditional_edges(
    "VERIFIER_RESET_THEN_CORRECTOR",
    route_after_verifier_reset,
    {
        "CORRECTOR": "CORRECTOR",
    },
)

graph.add_conditional_edges(
    "ENTRY_STATUS_CAPTURE",
    route_after_entry_status_capture,
    {
        "ARCHIVIST_QUEUE_ASSEMBLY": "ARCHIVIST_QUEUE_ASSEMBLY",
    },
)

graph.add_conditional_edges(
    "ARCHIVIST_QUEUE_ASSEMBLY",
    route_after_archivist_queue_assembly,
    {
        "ARCHIVIST_A1": "ARCHIVIST_A1",
        END: END,
    },
)

graph.add_conditional_edges(
    "ARCHIVIST_A1",
    route_after_archivist_a1,
    {
        "VALIDATOR": "VALIDATOR",
        "ARCHIVIST_A1": "ARCHIVIST_A1",
        "HIR_HALT": "HIR_HALT",
    },
)

graph.add_conditional_edges(
    "ARCHIVIST_A2",
    route_after_archivist_a2,
    {
        "VALIDATOR": "VALIDATOR",
        "ARCHIVIST_A2": "ARCHIVIST_A2",
        "A2_FEEDBACK_TO_A1": "A2_FEEDBACK_TO_A1",
        "HIR_HALT": "HIR_HALT",
    },
)

graph.add_conditional_edges(
    "A2_FEEDBACK_TO_A1",
    route_after_a2_feedback,
    {
        "ARCHIVIST_A1": "ARCHIVIST_A1",
        "HIR_HALT": "HIR_HALT",
    },
)

graph.add_conditional_edges(
    "VALIDATOR",
    route_after_validator,
    {
        "HARD_RESET": "HARD_RESET",
        "CORRECTOR": "CORRECTOR",
        "ARCHIVIST_A2": "ARCHIVIST_A2",
        "VALIDATOR_FEEDBACK_TO_A1": "VALIDATOR_FEEDBACK_TO_A1",
        "VALIDATOR_FEEDBACK_TO_A2": "VALIDATOR_FEEDBACK_TO_A2",
        "HIR_HALT": "HIR_HALT",
        END: END,
    },
)

graph.add_conditional_edges(
    "VALIDATOR_FEEDBACK_TO_A1",
    route_after_validator_feedback_a1,
    {
        "ARCHIVIST_A1": "ARCHIVIST_A1",
    },
)

graph.add_conditional_edges(
    "VALIDATOR_FEEDBACK_TO_A2",
    route_after_validator_feedback_a2,
    {
        "ARCHIVIST_A2": "ARCHIVIST_A2",
    },
)

graph.add_conditional_edges(
    "HARD_RESET",
    route_after_hard_reset,
    {
        "ARCHITECT": "ARCHITECT",
    },
)

# HIR_HALT is terminal
graph.add_edge("HIR_HALT", END)

# --- Compile ---
# NOTE: recursion_limit is passed at runtime via app.invoke(config={"recursion_limit": RECURSION_LIMIT})
# LangGraph 0.6.x does not accept recursion_limit at compile time.
app = graph.compile()

logger.info(
    "Orchestral Machine graph compiled (recursion_limit=%d at runtime, nodes=%d)",
    RECURSION_LIMIT,
    17,
)
