"""Orchestral Machine — Graph Routing Functions.

Conditional routing logic and feedback nodes for the LangGraph state machine.
"""

import logging
from typing import Any, Dict

from langgraph.graph import END

from src.config import (
    ARCHIVAL_ENABLED,
    MAX_ARCHIVAL_ATTEMPTS,
    MAX_CORRECTION_ATTEMPTS,
    MAX_LOOP_ITERATIONS,
    MAX_META_ATTEMPTS,
    MAX_VALIDATION_ATTEMPTS,
)
from src.graph_utils import _ensure_dict, _make_system_audit_entry, _now

logger = logging.getLogger(__name__)

_LINEAR_RESUME_PATH = [
    "ARCHITECT",
    "CODER",
    "REVIEWER",
    "TESTER",
    "CORRECTOR",
    "VERIFIER",
]


def _next_toward_resume_target(current_node: str, target: str) -> str:
    """During resume, route linearly toward the target node."""
    try:
        current_idx = _LINEAR_RESUME_PATH.index(current_node)
        target_idx = _LINEAR_RESUME_PATH.index(target)
    except ValueError:
        logger.error(
            "Resume routing: unknown node %s or target %s", current_node, target
        )
        return "HIR_HALT"
    if current_idx < target_idx:
        return _LINEAR_RESUME_PATH[current_idx + 1]
    logger.warning(
        "Resume routing: current %s >= target %s, halting", current_node, target
    )
    return "HIR_HALT"


def route_after_architect(state: dict) -> str:
    """ARCHITECT → CODER (always, on plan_ready)."""
    state = _ensure_dict(state)
    resume_target = state.get("resume_target")
    if resume_target:
        return _next_toward_resume_target("ARCHITECT", resume_target)
    status = state.get("status", "")
    if status == "HIR":
        return "HIR_HALT"
    # plan_ready → CODER
    return "CODER"


def route_after_coder(state: dict) -> str:
    """CODER → REVIEWER (always, on generated)."""
    state = _ensure_dict(state)
    resume_target = state.get("resume_target")
    if resume_target:
        return _next_toward_resume_target("CODER", resume_target)
    status = state.get("status", "")
    if status == "HIR":
        return "HIR_HALT"
    # generated → REVIEWER
    return "REVIEWER"


def route_after_reviewer(state: dict) -> str:
    """REVIEWER routing:
    - OK → TESTER
    - error_L1 → CORRECTOR
    - error_L2 → CORRECTOR
    - error_L3 → VERIFIER
    """
    state = _ensure_dict(state)
    resume_target = state.get("resume_target")
    if resume_target:
        return _next_toward_resume_target("REVIEWER", resume_target)
    status = state.get("status", "")

    if status == "HIR":
        return "HIR_HALT"
    if status == "OK":
        return "TESTER"
    if status in ("error_L1", "error_L2"):
        return "CORRECTOR"
    if status == "error_L3":
        return "VERIFIER"

    logger.error("Unexpected REVIEWER status: %s. Routing to HIR.", status)
    return "HIR_HALT"


def route_after_tester(state: dict) -> str:
    """TESTER routing:
    - PASS → ENTRY_STATUS_CAPTURE (then Archival Flow)
    - FAIL → CORRECTOR
    """
    state = _ensure_dict(state)
    resume_target = state.get("resume_target")
    if resume_target:
        return _next_toward_resume_target("TESTER", resume_target)
    status = state.get("status", "")

    if status == "HIR":
        return "HIR_HALT"
    if status == "PASS":
        return "ENTRY_STATUS_CAPTURE"
    if status == "FAIL":
        return "CORRECTOR"

    logger.error("Unexpected TESTER status: %s. Routing to HIR.", status)
    return "HIR_HALT"


def route_after_corrector(state: dict) -> str:
    """CORRECTOR routing:
    - fixed → REVIEWER
    - NEEDS_REWRITE → VERIFIER
    - no_change → REVIEWER (or VERIFIER if correction_attempt > MAX)
    """
    state = _ensure_dict(state)
    resume_target = state.get("resume_target")
    if resume_target:
        return _next_toward_resume_target("CORRECTOR", resume_target)
    status = state.get("status", "")
    correction_attempt = state.get("correction_attempt", 0)

    if status == "HIR":
        return "HIR_HALT"
    if status == "fixed":
        return "REVIEWER"
    if status == "NEEDS_REWRITE":
        return "VERIFIER"
    if status == "no_change":
        if correction_attempt > MAX_CORRECTION_ATTEMPTS:
            return "VERIFIER"
        return "REVIEWER"

    logger.error("Unexpected CORRECTOR status: %s. Routing to HIR.", status)
    return "HIR_HALT"


def route_after_verifier(state: dict) -> str:
    """VERIFIER routing (SPEC-18 one-time reset):
    - execution_failure → CORRECTOR (with one-time correction_attempt reset)
    - confirmed_L3 → ENTRY_STATUS_CAPTURE (Archival Flow)
    - confirmed_needs_rewrite → ENTRY_STATUS_CAPTURE (Archival Flow)
    - disagree_L3 → CORRECTOR
    """
    state = _ensure_dict(state)
    resume_target = state.get("resume_target")
    if resume_target:
        return _next_toward_resume_target("VERIFIER", resume_target)
    status = state.get("status", "")

    if status == "HIR":
        return "HIR_HALT"

    if status == "execution_failure":
        # One-time reset guard (SPEC-18)
        verifier_reset_used = state.get("verifier_reset_used", False)
        if not verifier_reset_used:
            logger.info(
                "Verifier execution_failure: applying one-time correction_attempt reset",
            )
            return "VERIFIER_RESET_THEN_CORRECTOR"
        else:
            logger.warning(
                "Verifier execution_failure: reset already used this loop. "
                "Escalating to HIR.",
            )
            return "HIR_HALT"

    if status in ("confirmed_L3", "confirmed_needs_rewrite"):
        return "ENTRY_STATUS_CAPTURE"

    if status == "disagree_L3":
        return "CORRECTOR"

    logger.error("Unexpected VERIFIER status: %s. Routing to HIR.", status)
    return "HIR_HALT"


def verifier_reset_node(state: dict) -> dict:
    """System node — applies one-time Verifier correction_attempt reset (SPEC-18)."""
    state = _ensure_dict(state)
    now = _now()
    task_id = state.get("task_id", "unknown")

    logger.info(
        "Verifier one-time reset: correction_attempt → 0, verifier_reset_used → True"
    )

    return {
        "correction_attempt": 0,
        "verifier_reset_used": True,
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "verifier_correction_reset",
                "detail": "One-time correction_attempt reset by Verifier (execution_failure)",
                "timestamp": now,
            }
        ],
        "audit_log": state.get("audit_log", [])
        + [
            _make_system_audit_entry(
                status="verifier_reset_applied",
                prompt_summary="One-time correction_attempt reset: 0, verifier_reset_used=True",
                task_id=task_id,
            ),
        ],
        "updated_at": now,
    }


def route_after_entry_status_capture(state: dict) -> str:
    """After capturing entry_status → ARCHIVIST_QUEUE_ASSEMBLY."""
    state = _ensure_dict(state)
    return "ARCHIVIST_QUEUE_ASSEMBLY"


def route_after_archivist_queue_assembly(state: dict) -> str:
    """After assembling queue → ARCHIVIST_A1, or END if archival is disabled."""
    state = _ensure_dict(state)
    if not ARCHIVAL_ENABLED:
        return END
    return "ARCHIVIST_A1"


def route_after_archivist_a1(state: dict) -> str:
    """ARCHIVIST_A1 routing:
    - archived → VALIDATOR
    - archival_error → retry A1 or HIR
    """
    state = _ensure_dict(state)
    status = state.get("status", "")
    archival_attempt = state.get("archival_attempt", 0)

    if status == "HIR":
        return "HIR_HALT"

    if status == "archived":
        return "VALIDATOR"

    if status == "archival_error":
        if archival_attempt < MAX_ARCHIVAL_ATTEMPTS:
            logger.warning(
                "Archivist_A1 archival_error (attempt %d/%d). Retrying.",
                archival_attempt,
                MAX_ARCHIVAL_ATTEMPTS,
            )
            return "ARCHIVIST_A1"
        else:
            logger.error(
                "Archivist_A1 archival_error: max attempts (%d) exhausted. HIR.",
                MAX_ARCHIVAL_ATTEMPTS,
            )
            return "HIR_HALT"

    logger.error("Unexpected ARCHIVIST_A1 status: %s. Routing to HIR.", status)
    return "HIR_HALT"


def route_after_archivist_a2(state: dict) -> str:
    """ARCHIVIST_A2 routing:
    - archived_meta_summary → VALIDATOR
    - meta_summary_error → retry A2 or HIR
    - archived_summary_corrupt → A1 with feedback
    """
    state = _ensure_dict(state)
    status = state.get("status", "")
    meta_attempt = state.get("meta_attempt", 0)

    if status == "HIR":
        return "HIR_HALT"

    if status == "archived_meta_summary":
        return "VALIDATOR"

    if status == "meta_summary_error":
        if meta_attempt < MAX_META_ATTEMPTS:
            logger.warning(
                "Archivist_A2 meta_summary_error (attempt %d/%d). Retrying.",
                meta_attempt,
                MAX_META_ATTEMPTS,
            )
            return "ARCHIVIST_A2"
        else:
            logger.error(
                "Archivist_A2 meta_summary_error: max attempts (%d) exhausted. HIR.",
                MAX_META_ATTEMPTS,
            )
            return "HIR_HALT"

    if status == "archived_summary_corrupt":
        # Downgrade: send back to A1 with feedback from A2
        logger.warning(
            "Archivist_A2 reports archived_summary_corrupt. "
            "Routing back to ARCHIVIST_A1 with feedback.",
        )
        return "A2_FEEDBACK_TO_A1"

    logger.error("Unexpected ARCHIVIST_A2 status: %s. Routing to HIR.", status)
    return "HIR_HALT"


def a2_feedback_to_a1_node(state: dict) -> dict:
    """System node — extracts A2 correction_instructions and routes to A1 (SPEC-12)."""
    state = _ensure_dict(state)
    now = _now()
    correction_instructions = state.get("correction_instructions") or []
    if not correction_instructions:
        meta_summary = state.get("meta_summary") or {}
        correction_instructions = meta_summary.get("correction_instructions") or []
    if not correction_instructions:
        correction_instructions = [
            "Archived summary reported as corrupt by A2. Please re-archive."
        ]

    logger.info("Populating archivist_feedback from A2 correction_instructions")

    return {
        "archivist_feedback": correction_instructions,
        "archived_meta_summary_ref": None,
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "a2_feedback_to_a1",
                "detail": f"A2 sent {len(correction_instructions)} correction instructions to A1",
                "timestamp": now,
            }
        ],
        "updated_at": now,
    }


def route_after_a2_feedback(state: dict) -> str:
    """After A2 feedback population → ARCHIVIST_A1, with attempt guard."""
    state = _ensure_dict(state)
    archival_attempt = state.get("archival_attempt", 0)
    if archival_attempt >= MAX_ARCHIVAL_ATTEMPTS:
        logger.error(
            "A2 feedback loop: archival_attempt=%d >= MAX_ARCHIVAL_ATTEMPTS=%d. HIR.",
            archival_attempt,
            MAX_ARCHIVAL_ATTEMPTS,
        )
        return "HIR_HALT"
    return "ARCHIVIST_A1"


def route_after_validator(state: dict) -> str:
    """VALIDATOR routing (SPEC-11, SPEC-12, SPEC-19, SPEC-20):
    - ready_for_next → route by entry_status
    - approved_hard_reset → HARD_RESET
    - COMPLETED → END
    - HIR → HIR_HALT
    - unapproved_archived_summary → A1 with feedback
    - unapproved_meta_summary → A2 with feedback
    """
    state = _ensure_dict(state)
    status = state.get("status", "")
    entry_status = state.get("entry_status", "")
    validation_attempt = state.get("validation_attempt", 0)
    archival_attempt = state.get("archival_attempt", 0)
    loop_iteration = state.get("loop_iteration", 0)

    if status == "HIR":
        return "HIR_HALT"

    if status == "COMPLETED":
        return END

    if status == "approved_hard_reset":
        # Check loop_iteration limit before allowing reset
        if loop_iteration >= MAX_LOOP_ITERATIONS - 1:
            logger.error(
                "approved_hard_reset requested but loop_iteration=%d "
                "would exceed max=%d. HIR.",
                loop_iteration,
                MAX_LOOP_ITERATIONS,
            )
            return "HIR_HALT"
        return "HARD_RESET"

    if status == "unapproved_archived_summary":
        if validation_attempt >= MAX_VALIDATION_ATTEMPTS:
            logger.error(
                "Validator rejection limit exhausted (validation_attempt=%d). HIR.",
                validation_attempt,
            )
            return "HIR_HALT"
        return "VALIDATOR_FEEDBACK_TO_A1"

    if status == "unapproved_meta_summary":
        if validation_attempt >= MAX_VALIDATION_ATTEMPTS:
            logger.error(
                "Validator rejection limit exhausted (validation_attempt=%d). HIR.",
                validation_attempt,
            )
            return "HIR_HALT"
        return "VALIDATOR_FEEDBACK_TO_A2"

    if status == "ready_for_next":
        # Check if this was a validated_archived_summary or validated_meta_summary

        # Determine if meta-summary is needed/done
        archived_meta_ref = state.get("archived_meta_summary_ref")
        has_meta = archived_meta_ref is not None

        # If no meta-summary yet and entry_status requires it
        if not has_meta:
            # PASS always requires meta; also when archival_attempt > 4
            if entry_status == "PASS" or archival_attempt > MAX_ARCHIVAL_ATTEMPTS:
                return "ARCHIVIST_A2"
            # For error statuses, route back to correction without meta
            if entry_status in ("FAIL", "error_L1", "error_L2"):
                return "CORRECTOR"
            # For confirmed statuses that need hard reset path
            if entry_status in ("confirmed_needs_rewrite", "confirmed_L3"):
                return "ARCHIVIST_A2"

        # Meta exists and was validated → final routing by entry_status
        if has_meta:
            if entry_status == "PASS":
                return END
            if entry_status in ("confirmed_needs_rewrite", "confirmed_L3"):
                # These require hard reset — Validator should have emitted approved_hard_reset
                # but if we reach here with ready_for_next, it means approved
                return "HARD_RESET"
            if entry_status in ("FAIL", "error_L1", "error_L2"):
                return "CORRECTOR"

        # Safety: loop_iteration > 1 → HIR
        if loop_iteration > 1:
            logger.error(
                "Safety limit: loop_iteration=%d > 1 with ready_for_next. HIR.",
                loop_iteration,
            )
            return "HIR_HALT"

        logger.error(
            "Unexpected ready_for_next routing: entry_status=%s. HIR.",
            entry_status,
        )
        return "HIR_HALT"

    logger.error("Unexpected VALIDATOR status: %s. Routing to HIR.", status)
    return "HIR_HALT"


def validator_feedback_to_a1_node(state: dict) -> dict:
    """System node — extracts Validator correction_instructions for A1 retry (SPEC-12)."""
    state = _ensure_dict(state)
    now = _now()
    correction_instructions = state.get("correction_instructions", [])
    if not correction_instructions:
        correction_instructions = [
            "Validator rejected archived summary. Please re-archive with corrections."
        ]

    new_validation_attempt = state.get("validation_attempt", 0) + 1

    logger.info(
        "Populating archivist_feedback from Validator for A1 retry "
        "(validation_attempt=%d)",
        new_validation_attempt,
    )

    return {
        "archivist_feedback": correction_instructions,
        "validation_attempt": new_validation_attempt,
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "validator_feedback_to_a1",
                "detail": (
                    f"Validator sent {len(correction_instructions)} correction instructions "
                    f"to A1 (validation_attempt={new_validation_attempt})"
                ),
                "timestamp": now,
            }
        ],
        "updated_at": now,
    }


def validator_feedback_to_a2_node(state: dict) -> dict:
    """System node — extracts Validator correction_instructions for A2 retry (SPEC-12)."""
    state = _ensure_dict(state)
    now = _now()
    correction_instructions = state.get("correction_instructions", [])
    if not correction_instructions:
        correction_instructions = [
            "Validator rejected meta-summary. Please regenerate with corrections."
        ]

    new_validation_attempt = state.get("validation_attempt", 0) + 1
    new_meta_attempt = state.get("meta_attempt", 0) + 1

    logger.info(
        "Populating archivist_feedback from Validator for A2 retry "
        "(validation_attempt=%d, meta_attempt=%d)",
        new_validation_attempt,
        new_meta_attempt,
    )

    return {
        "archivist_feedback": correction_instructions,
        "validation_attempt": new_validation_attempt,
        "meta_attempt": new_meta_attempt,
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "validator_feedback_to_a2",
                "detail": (
                    f"Validator sent {len(correction_instructions)} correction instructions "
                    f"to A2 (validation_attempt={new_validation_attempt}, "
                    f"meta_attempt={new_meta_attempt})"
                ),
                "timestamp": now,
            }
        ],
        "updated_at": now,
    }


def route_after_validator_feedback_a1(state: dict) -> str:
    """After Validator feedback for A1 → ARCHIVIST_A1."""
    state = _ensure_dict(state)
    return "ARCHIVIST_A1"


def route_after_validator_feedback_a2(state: dict) -> str:
    """After Validator feedback for A2 → ARCHIVIST_A2."""
    state = _ensure_dict(state)
    return "ARCHIVIST_A2"


def route_after_hard_reset(state: dict) -> str:
    """After Hard Reset → ARCHITECT (always)."""
    state = _ensure_dict(state)
    return "ARCHITECT"


def route_after_verifier_reset(state: dict) -> str:
    """After Verifier one-time reset → CORRECTOR."""
    state = _ensure_dict(state)
    return "CORRECTOR"
