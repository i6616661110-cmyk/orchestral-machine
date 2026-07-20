"""Orchestral Machine — Control Interface Module.

Single source of truth for all operator control logic.  CLI and API layers
(steps 16–17) delegate to public functions in this module exclusively.

Implements:
    §10.1  Operator Commands  — HALT, RESUME, FORCE_RESET
    CHECKPOINT_RESUME_SYSTEM §4 Mode 2 — RESUME_REVERT
    §10 L839-844  Debug Hooks — dry_run, inject_state, simulate_event
    Checkpoint Management — delegation to src.checkpoint

Functions:
    operator_halt              — halt system, save checkpoint
    operator_resume            — resume from checkpoint (continue)
    operator_resume_revert     — resume with rollback to target node
    operator_force_reset       — confirmed hard reset with archival snapshot
    get_system_status          — combined metadata + state status query
    checkpoint_create          — manual checkpoint creation
    checkpoint_list            — list available checkpoints
    checkpoint_inspect         — load and return checkpoint contents
    checkpoint_validate        — validate checkpoint integrity
    checkpoint_clean           — clean old checkpoints
    dry_run                    — mock-execute node (dev only)
    inject_state               — merge partial state (dev only)
    simulate_event             — inject synthetic event (dev only)
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.checkpoint import (
    CheckpointCorruptedError,
    cleanup_old_checkpoints,
    create_checkpoint,
    get_latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    rollback_state_to_node,
    validate_checkpoint,
)
from src.config import MAX_LOOP_ITERATIONS
from src.graph import _controller_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class InvalidStateError(Exception):
    """Raised when an operator command is issued in an invalid system state."""


# ---------------------------------------------------------------------------
# Environment Guard
# ---------------------------------------------------------------------------

_IS_PRODUCTION = os.getenv("ORCHESTRAL_ENV", "development") == "production"


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _operator_audit_entry(
    *,
    status: str,
    decision_type: str = "operator_command",
    input_hash: Optional[str] = None,
    output_hash: Optional[str] = None,
) -> dict:
    """Build audit entry with source='operator' for constitutional exception."""
    return {
        "node": "GRAPH_CONTROLLER",
        "model_id": "operator",
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": status,
        "decision_type": decision_type,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "seed": None,
        "source": "operator",
    }


# ---------------------------------------------------------------------------
# COMMAND: HALT (§10.1 L848-897)
# ---------------------------------------------------------------------------


def operator_halt(state: dict) -> dict:
    """Halt the system immediately and persist a checkpoint.

    Preconditions: None (HALT is always allowed).

    Effects:
        1. Set system_state to HALTED.
        2. Create checkpoint with reason=OPERATOR_HALT.
        3. Append operator audit entry with status=HALTED.
        4. Set system_flags.checkpoint_saved = True.

    Args:
        state: Current flat GraphState dict.

    Returns:
        Result dict with status, checkpoint path, and message.
    """
    # 1. Set system_state
    _controller_metadata["system_state"] = "HALTED"

    # 2. Create checkpoint
    checkpoint_path = create_checkpoint(
        graph_state=state,
        reason="OPERATOR_HALT",
        last_completed_node=_controller_metadata.get("last_completed_node"),
        next_scheduled_node=_controller_metadata.get("next_scheduled_node"),
        system_state=_controller_metadata["system_state"],
        recursion_count=state.get("recursion_count", 0),
    )

    # 3. Append operator audit entry
    audit_log: list = list(state.get("audit_log", []))
    audit_log.append(_operator_audit_entry(status="HALTED"))
    state["audit_log"] = audit_log

    # 4. Set checkpoint_saved flag
    flags: dict = dict(state.get("system_flags", {}))
    flags["checkpoint_saved"] = True
    state["system_flags"] = flags

    # 5. Log
    logger.critical(
        "System HALTED by operator. Checkpoint saved: %s",
        checkpoint_path,
    )

    return {
        "status": "HALTED",
        "checkpoint": checkpoint_path,
        "message": "System halted successfully. Use RESUME to continue.",
    }


# ---------------------------------------------------------------------------
# COMMAND: RESUME (§10.1 L900-955)
# ---------------------------------------------------------------------------


def operator_resume(
    state: dict,
    checkpoint_path: Optional[str] = None,
) -> dict:
    """Resume execution from a checkpoint (continue mode).

    Preconditions:
        system_state must be HALTED.

    Effects:
        1. Load checkpoint (from path or latest).
        2. Validate checkpoint integrity.
        3. Merge checkpoint graph_state into state.
        4. Restore controller metadata from checkpoint.
        5. Set system_state to RUNNING.
        6. Append operator audit entry with status=RESUMED.

    Args:
        state: Current flat GraphState dict.
        checkpoint_path: Optional explicit checkpoint file path.

    Returns:
        Result dict with status, resumed_from, and next_node.

    Raises:
        InvalidStateError: If system_state is not HALTED.
        FileNotFoundError: If no checkpoint is found.
        CheckpointCorruptedError: If checkpoint fails integrity validation.
    """
    # Precondition
    if _controller_metadata["system_state"] != "HALTED":
        raise InvalidStateError(
            f"RESUME requires system_state=HALTED, "
            f"got '{_controller_metadata['system_state']}'"
        )

    # 1. Load checkpoint
    path = checkpoint_path or get_latest_checkpoint()
    if path is None:
        raise FileNotFoundError("No checkpoint found for RESUME")

    checkpoint = load_checkpoint(path)

    # 2. Validate integrity
    if not validate_checkpoint(checkpoint):
        raise CheckpointCorruptedError(
            f"Checkpoint integrity validation failed: {path}"
        )

    # 3. Extract graph_state and merge into state
    graph_state: dict = checkpoint.get("graph_state", {})
    state.update(deepcopy(graph_state))

    # 4. Restore controller metadata
    checkpoint_metadata: dict = checkpoint.get("checkpoint_metadata", {})
    _controller_metadata["system_state"] = "RUNNING"
    _controller_metadata["last_completed_node"] = checkpoint_metadata.get(
        "last_completed_node",
        _controller_metadata.get("last_completed_node"),
    )
    _controller_metadata["next_scheduled_node"] = checkpoint_metadata.get(
        "next_scheduled_node",
        _controller_metadata.get("next_scheduled_node"),
    )

    # 5. Append operator audit entry
    audit_log: list = list(state.get("audit_log", []))
    audit_log.append(_operator_audit_entry(status="RESUMED"))
    state["audit_log"] = audit_log

    logger.info(
        "System RESUMED from checkpoint: %s (next_node=%s)",
        path,
        checkpoint_metadata.get("next_scheduled_node"),
    )

    return {
        "status": "RESUMED",
        "resumed_from": checkpoint_metadata.get("last_completed_node"),
        "next_node": checkpoint_metadata.get("next_scheduled_node"),
    }


# ---------------------------------------------------------------------------
# COMMAND: RESUME_REVERT (CHECKPOINT_RESUME_SYSTEM §4 Mode 2)
# ---------------------------------------------------------------------------


def operator_resume_revert(
    state: dict,
    target_node: str,
    checkpoint_path: Optional[str] = None,
) -> dict:
    """Resume with rollback to a specific target node.

    Preconditions:
        system_state must be HALTED.

    Effects:
        1. Load and validate checkpoint.
        2. Extract graph_state from checkpoint.
        3. Apply rollback_state_to_node for target_node.
        4. Merge rolled-back state into state.
        5. Set system_state to RUNNING.
        6. Update next_scheduled_node to target_node.
        7. Append operator audit entry with status=REVERTED.

    Args:
        state: Current flat GraphState dict.
        target_node: Node to revert execution back to.
        checkpoint_path: Optional explicit checkpoint file path.

    Returns:
        Result dict with status, target_node, and resumed_from.

    Raises:
        InvalidStateError: If system_state is not HALTED.
        FileNotFoundError: If no checkpoint is found.
        CheckpointCorruptedError: If checkpoint fails integrity validation.
        ValueError: If target_node is not a supported rollback target.
    """
    # Precondition
    if _controller_metadata["system_state"] != "HALTED":
        raise InvalidStateError(
            f"RESUME_REVERT requires system_state=HALTED, "
            f"got '{_controller_metadata['system_state']}'"
        )

    # 1. Load checkpoint
    path = checkpoint_path or get_latest_checkpoint()
    if path is None:
        raise FileNotFoundError("No checkpoint found for RESUME_REVERT")

    checkpoint = load_checkpoint(path)

    # 2. Validate integrity
    if not validate_checkpoint(checkpoint):
        raise CheckpointCorruptedError(
            f"Checkpoint integrity validation failed: {path}"
        )

    # 3. Extract graph_state
    graph_state: dict = checkpoint.get("graph_state", {})

    # 4. Apply rollback
    rolled_back = rollback_state_to_node(graph_state, target_node)

    # 5. Merge rolled-back state into state
    state.update(deepcopy(rolled_back))

    # 6. Update controller metadata
    checkpoint_metadata: dict = checkpoint.get("checkpoint_metadata", {})
    _controller_metadata["system_state"] = "RUNNING"
    _controller_metadata["next_scheduled_node"] = target_node

    # 7. Append operator audit entry
    audit_log: list = list(state.get("audit_log", []))
    audit_log.append(
        _operator_audit_entry(
            status="REVERTED",
            decision_type="operator_resume_revert",
        )
    )
    state["audit_log"] = audit_log

    logger.info(
        "System REVERTED to node %s from checkpoint: %s",
        target_node,
        path,
    )

    return {
        "status": "REVERTED",
        "target_node": target_node,
        "resumed_from": checkpoint_metadata.get("last_completed_node"),
    }


# ---------------------------------------------------------------------------
# COMMAND: FORCE_RESET (§10.1 L958-1060)
# ---------------------------------------------------------------------------


def operator_force_reset(
    state: dict,
    confirmation: bool = False,
) -> dict:
    """Execute a confirmed hard reset with archival snapshot.

    Preconditions:
        system_state must be HALTED or HIR.

    Phase 1 — loop_iteration check:
        If loop_iteration > 1, return RESET_DENIED (does NOT raise).

    Phase 2 — confirmation check:
        If confirmation is False, return CONFIRMATION_REQUIRED with summary.

    Phase 3 — execute reset:
        1. Create archival checkpoint (reason=OPERATOR_FORCE_RESET).
        2. Apply reset mutations atomically.
        3. Set system_state to RUNNING.
        4. Set next_scheduled_node to ARCHITECT.
        5. Append operator audit entry with status=FORCE_RESET.

    Args:
        state: Current flat GraphState dict.
        confirmation: Explicit operator confirmation for destructive action.

    Returns:
        Result dict with status and relevant metadata.

    Raises:
        InvalidStateError: If system_state is not HALTED or HIR.
    """
    current_system_state = _controller_metadata["system_state"]

    # Precondition
    if current_system_state not in ("HALTED", "HIR"):
        raise InvalidStateError(
            f"FORCE_RESET requires system_state=HALTED or HIR, "
            f"got '{current_system_state}'"
        )

    # Phase 1: loop_iteration check
    loop_iteration = state.get("loop_iteration", 0)
    if loop_iteration > 1:
        logger.warning(
            "FORCE_RESET denied: loop_iteration=%d exceeds limit (MAX_LOOP_ITERATIONS=%d)",
            loop_iteration,
            MAX_LOOP_ITERATIONS,
        )
        return {
            "status": "RESET_DENIED",
            "message": (
                f"FORCE_RESET denied: loop_iteration ({loop_iteration}) "
                f"exceeds maximum ({MAX_LOOP_ITERATIONS}). "
                f"The system has already exhausted its rewrite budget."
            ),
        }

    # Phase 2: confirmation check
    if not confirmation:
        return {
            "status": "CONFIRMATION_REQUIRED",
            "message": (
                "FORCE_RESET is a destructive operation that clears plan, "
                "code, and all correction state. An archival checkpoint will "
                "be saved before reset. Pass confirmation=True to proceed."
            ),
            "current_state_summary": {
                "system_state": current_system_state,
                "loop_iteration": loop_iteration,
                "correction_attempt": state.get("correction_attempt", 0),
                "status": state.get("status"),
                "task_id": state.get("task_id"),
                "last_completed_node": _controller_metadata.get("last_completed_node"),
            },
        }

    # Phase 3: execute reset

    # 3.1 Create archival checkpoint
    checkpoint_path = create_checkpoint(
        graph_state=state,
        reason="OPERATOR_FORCE_RESET",
        last_completed_node=_controller_metadata.get("last_completed_node"),
        next_scheduled_node=_controller_metadata.get("next_scheduled_node"),
        system_state=current_system_state,
        recursion_count=state.get("recursion_count", 0),
    )

    # 3.2 Apply reset mutations atomically
    state["plan"] = None
    state["code"] = {}
    state["correction_attempt"] = 0
    state["archival_attempt"] = 0
    state["loop_iteration"] = loop_iteration + 1
    state["verifier_reset_used"] = False
    state["status"] = "rewrite_confirmed"

    # 3.3 Set controller metadata
    _controller_metadata["system_state"] = "RUNNING"
    _controller_metadata["next_scheduled_node"] = "ARCHITECT"

    # 3.4 Append operator audit entry
    audit_log: list = list(state.get("audit_log", []))
    audit_log.append(_operator_audit_entry(status="FORCE_RESET"))
    state["audit_log"] = audit_log

    logger.critical(
        "FORCE_RESET executed by operator. Archival snapshot: %s. "
        "loop_iteration now %d. Routing to ARCHITECT.",
        checkpoint_path,
        state["loop_iteration"],
    )

    return {
        "status": "RESET_COMPLETE",
        "snapshot_ref": checkpoint_path,
        "new_loop_iteration": state["loop_iteration"],
    }


# ---------------------------------------------------------------------------
# SYSTEM STATUS QUERY
# ---------------------------------------------------------------------------


def get_system_status(state: dict) -> dict:
    """Return combined system status from controller metadata and state.

    Args:
        state: Current flat GraphState dict.

    Returns:
        Dict with system_state, node progression, counters, and task info.
    """
    return {
        "system_state": _controller_metadata["system_state"],
        "last_completed_node": _controller_metadata["last_completed_node"],
        "next_scheduled_node": _controller_metadata["next_scheduled_node"],
        "loop_iteration": state.get("loop_iteration", 0),
        "correction_attempt": state.get("correction_attempt", 0),
        "status": state.get("status"),
        "task_id": state.get("task_id"),
    }


# ---------------------------------------------------------------------------
# DEBUG HOOKS (§10 L839-844)
# ---------------------------------------------------------------------------


def dry_run(node_name: str, state: dict) -> dict:
    """Execute a node with mock responses to verify routing logic.

    Simulates a successful node completion without invoking the LLM,
    returning a mock state update that matches the expected output
    structure of the given node.

    Disabled in production.

    Args:
        node_name: Name of the node to simulate (e.g. "ARCHITECT").
        state: Current flat GraphState dict.

    Returns:
        Mock state update dict simulating the node completed successfully.

    Raises:
        RuntimeError: If called in production mode.
    """
    if _IS_PRODUCTION:
        raise RuntimeError("dry_run is disabled in production mode")

    timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # Build a generic mock output reflecting the node completed
    mock_output: Dict[str, Any] = {
        "status": "mock_complete",
        "node": node_name,
        "dry_run": True,
        "timestamp": timestamp,
    }

    # Append audit entry for traceability
    audit_log: list = list(state.get("audit_log", []))
    audit_log.append(
        {
            "node": node_name,
            "model_id": "dry_run",
            "timestamp": timestamp,
            "status": "mock_complete",
            "decision_type": "dry_run",
            "input_hash": None,
            "output_hash": None,
            "seed": None,
            "source": "operator",
        }
    )
    mock_output["audit_log"] = audit_log

    logger.info("DRY RUN executed for node %s", node_name)

    return mock_output


def inject_state(state: dict, partial_state: dict) -> dict:
    """Load a partial state snapshot to test specific scenarios.

    Merges the partial_state dict into the current state.
    Useful for setting up specific test conditions without running
    the full pipeline.

    Disabled in production.

    Args:
        state: Current flat GraphState dict (modified in place).
        partial_state: Dict of state fields to merge.

    Returns:
        Updated state dict after merge.

    Raises:
        RuntimeError: If called in production mode.
    """
    if _IS_PRODUCTION:
        raise RuntimeError("inject_state is disabled in production mode")

    state.update(partial_state)

    # Append audit entry
    audit_log: list = list(state.get("audit_log", []))
    audit_log.append(
        _operator_audit_entry(
            status="state_injected",
            decision_type="debug_inject_state",
        )
    )
    state["audit_log"] = audit_log

    logger.info(
        "STATE INJECTED: %d fields merged into state",
        len(partial_state),
    )

    return state


def simulate_event(state: dict, event_name: str, payload: dict) -> dict:
    """Inject a synthetic event to test graph reactions.

    Appends the event to event_log and optionally applies payload
    fields to state, allowing simulation of specific event-driven
    routing scenarios.

    Disabled in production.

    Args:
        state: Current flat GraphState dict (modified in place).
        event_name: Name of the synthetic event (e.g. "test_failure").
        payload: Dict of state mutations to apply alongside the event.

    Returns:
        Updated state dict with the synthetic event appended.

    Raises:
        RuntimeError: If called in production mode.
    """
    if _IS_PRODUCTION:
        raise RuntimeError("simulate_event is disabled in production mode")

    timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # Apply payload to state
    state.update(payload)

    # Append event
    event_log: list = list(state.get("event_log", []))
    event_log.append(
        {
            "event": event_name,
            "detail": f"Simulated event with payload keys: {sorted(payload.keys())}",
            "timestamp": timestamp,
        }
    )
    state["event_log"] = event_log

    # Append audit entry
    audit_log: list = list(state.get("audit_log", []))
    audit_log.append(
        _operator_audit_entry(
            status="event_simulated",
            decision_type="debug_simulate_event",
        )
    )
    state["audit_log"] = audit_log

    logger.info(
        "SIMULATED EVENT: %s (payload keys: %s)",
        event_name,
        sorted(payload.keys()),
    )

    return state


# ---------------------------------------------------------------------------
# CHECKPOINT MANAGEMENT DELEGATION
# ---------------------------------------------------------------------------


def checkpoint_create(state: dict, label: Optional[str] = None) -> str:
    """Operator-initiated manual checkpoint creation.

    Args:
        state: Current flat GraphState dict.
        label: Optional human-readable label for the checkpoint file.

    Returns:
        Absolute path of the created checkpoint file.
    """
    path = create_checkpoint(
        graph_state=state,
        reason="OPERATOR_MANUAL",
        last_completed_node=_controller_metadata.get("last_completed_node"),
        next_scheduled_node=_controller_metadata.get("next_scheduled_node"),
        system_state=_controller_metadata["system_state"],
        recursion_count=state.get("recursion_count", 0),
        label=label,
    )

    # Append audit entry
    audit_log: list = list(state.get("audit_log", []))
    audit_log.append(_operator_audit_entry(status="CHECKPOINT_CREATED"))
    state["audit_log"] = audit_log

    logger.info("Manual checkpoint created: %s", path)

    return path


def checkpoint_list() -> list:
    """List all available checkpoints.

    Returns:
        Sorted list of checkpoint file path strings.
    """
    return list_checkpoints()


def checkpoint_inspect(path: str) -> dict:
    """Load and return checkpoint contents.

    Args:
        path: File system path to the checkpoint file.

    Returns:
        Parsed checkpoint dict.
    """
    return load_checkpoint(path)


def checkpoint_validate(path: str) -> dict:
    """Validate checkpoint integrity.

    Args:
        path: File system path to the checkpoint file.

    Returns:
        Dict with 'valid' boolean and 'path' string.
    """
    checkpoint = load_checkpoint(path)
    is_valid = validate_checkpoint(checkpoint)
    return {"valid": is_valid, "path": path}


def checkpoint_clean(keep: int = 10) -> dict:
    """Clean old checkpoints, retaining only the most recent *keep*.

    Args:
        keep: Number of most recent checkpoints to retain.

    Returns:
        Dict with 'deleted' count and 'kept' count.
    """
    deleted = cleanup_old_checkpoints(keep=keep)
    logger.info("Checkpoint cleanup: deleted=%d, kept=%d", deleted, keep)
    return {"deleted": deleted, "kept": keep}
