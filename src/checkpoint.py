"""Orchestral Machine — Checkpoint Persistence & Resume Module.

Handles creation, validation, loading, and management of execution
checkpoints for the LangGraph state machine. Supports three resume modes:

    RESUME_CONTINUE — reload and continue from next_scheduled_node
    RESUME_REVERT   — rollback state to an earlier safe node
    RESUME_RESET    — operator-initiated hard reset with archival snapshot

Checkpoint files are immutable JSON with SHA-256 integrity hashes,
stored in ``./checkpoints/``.

Functions:
    create_checkpoint         — serialize state + metadata + integrity hash
    load_checkpoint           — read checkpoint JSON from disk
    validate_checkpoint       — verify SHA-256 integrity
    list_checkpoints          — enumerate available checkpoint files
    get_latest_checkpoint     — return most recent checkpoint path
    cleanup_old_checkpoints   — retain only the *keep* most recent files
    rollback_state_to_node    — deterministic state cleanup for safe revert
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path("./checkpoints")


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class CheckpointCorruptedError(Exception):
    """Raised when a checkpoint fails SHA-256 integrity validation."""


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return current UTC timestamp in ISO-8601 format with 'Z' suffix."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _stable_json(data: Dict[str, Any]) -> str:
    """Produce deterministic JSON string for hashing.

    Uses sorted keys and compact separators to guarantee identical
    output for identical data, regardless of insertion order.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(payload: str) -> str:
    """Return prefixed SHA-256 hex digest of *payload*."""
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


# ---------------------------------------------------------------------------
# Checkpoint Creation
# ---------------------------------------------------------------------------


def create_checkpoint(
    *,
    graph_state: Dict[str, Any],
    reason: str,
    last_completed_node: Optional[str],
    next_scheduled_node: Optional[str],
    system_state: str,
    recursion_count: int,
    label: Optional[str] = None,
) -> str:
    """Create an immutable checkpoint file with integrity hash.

    Args:
        graph_state: Full flat ``GraphState`` dict snapshot.
        reason: Why the checkpoint was created (e.g. ``OPERATOR_HALT``).
        last_completed_node: Name of the last successfully completed node.
        next_scheduled_node: Name of the node scheduled to execute next.
        system_state: Current system state (``RUNNING``, ``HALTED``, ``HIR``).
        recursion_count: Current LangGraph recursion counter.
        label: Optional human-readable label appended to filename.

    Returns:
        Absolute string path of the written checkpoint file.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "checkpoint_metadata": {
            "created_at": _utc_now(),
            "task_id": graph_state.get("task_id", "unknown"),
            "reason": reason,
            "last_completed_node": last_completed_node,
            "next_scheduled_node": next_scheduled_node,
            "recursion_count": recursion_count,
            "system_state": system_state,
        },
        "graph_state": graph_state,
        "environment": {
            "python_version": platform.python_version(),
            "orchestral_env": os.getenv("ORCHESTRAL_ENV", "development"),
            "host": platform.node(),
        },
    }

    # Hash excludes the hash field itself.
    digest_input = _stable_json(checkpoint)
    checkpoint["checkpoint_hash"] = _sha256(digest_input)

    task_id = graph_state.get("task_id", "unknown")
    base = f"checkpoint_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if label:
        base = f"{base}_{label}"
    path = CHECKPOINT_DIR / f"{base}.json"

    path.write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Checkpoint created: %s (reason=%s, task_id=%s)",
        path,
        reason,
        task_id,
    )

    return str(path)


# ---------------------------------------------------------------------------
# Checkpoint Loading & Validation
# ---------------------------------------------------------------------------


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load and parse a checkpoint JSON file.

    Args:
        path: File system path to the checkpoint file.

    Returns:
        Parsed checkpoint dict with ``checkpoint_metadata``,
        ``graph_state``, ``environment``, and ``checkpoint_hash``.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = json.loads(p.read_text(encoding="utf-8"))
    logger.info("Checkpoint loaded: %s", path)
    return checkpoint


def validate_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    """Verify SHA-256 integrity of a checkpoint.

    Recomputes the hash from all fields except ``checkpoint_hash``
    and compares against the stored value.

    Args:
        checkpoint: Parsed checkpoint dict.

    Returns:
        ``True`` if the hash matches, ``False`` otherwise.
    """
    expected = checkpoint.get("checkpoint_hash")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        logger.warning("Checkpoint has missing or malformed checkpoint_hash")
        return False

    # Reconstruct the content without the hash field itself.
    content = dict(checkpoint)
    content.pop("checkpoint_hash", None)

    actual = _sha256(_stable_json(content))
    is_valid = actual == expected

    if not is_valid:
        logger.error(
            "Checkpoint integrity failure: expected=%s, actual=%s",
            expected,
            actual,
        )
    else:
        logger.info("Checkpoint integrity validated successfully")

    return is_valid


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------


def list_checkpoints(directory: str = "./checkpoints") -> List[str]:
    """Return sorted list of checkpoint file paths in *directory*.

    Args:
        directory: Path to the checkpoints directory.

    Returns:
        List of checkpoint file paths, sorted chronologically.
    """
    p = Path(directory)
    if not p.exists():
        return []
    return [str(x) for x in sorted(p.glob("checkpoint_*.json"))]


def get_latest_checkpoint(directory: str = "./checkpoints") -> Optional[str]:
    """Return the path of the most recent checkpoint, or ``None``.

    Args:
        directory: Path to the checkpoints directory.

    Returns:
        Path string of the latest checkpoint file, or ``None`` if none exist.
    """
    files = list_checkpoints(directory)
    return files[-1] if files else None


def cleanup_old_checkpoints(
    keep: int = 10,
    directory: str = "./checkpoints",
) -> int:
    """Delete old checkpoint files, retaining only the most recent *keep*.

    Args:
        keep: Number of most recent checkpoints to retain.
        directory: Path to the checkpoints directory.

    Returns:
        Number of checkpoint files deleted.
    """
    files = list_checkpoints(directory)
    if len(files) <= keep:
        return 0

    to_delete = files[: len(files) - keep]
    for fp in to_delete:
        Path(fp).unlink(missing_ok=True)
        logger.info("Deleted old checkpoint: %s", fp)

    logger.info(
        "Checkpoint cleanup: deleted %d, kept %d",
        len(to_delete),
        keep,
    )
    return len(to_delete)


# ---------------------------------------------------------------------------
# State Rollback (RESUME_REVERT)
# ---------------------------------------------------------------------------

# Safe revert paths matrix (derived from graph dependencies):
#
# | Current progression point          | Safe revert targets                                      |
# |------------------------------------|----------------------------------------------------------|
# | Before/at CODER                    | ARCHITECT, CODER                                         |
# | Before/at REVIEWER                 | CODER, REVIEWER                                          |
# | Before/at TESTER                   | REVIEWER, TESTER                                         |
# | In CORRECTOR loop                  | CORRECTOR_C1, CORRECTOR_C2, REVIEWER                     |
# | In VERIFIER path                   | VERIFIER, CORRECTOR_C1, REVIEWER                         |
# | In ARCHIVIST/VALIDATOR flow        | ARCHIVIST_A1, ARCHIVIST_A2, VALIDATOR, REVIEWER           |

_SUPPORTED_ROLLBACK_TARGETS = frozenset(
    {
        "ARCHITECT",
        "CODER",
        "REVIEWER",
        "TESTER",
        "CORRECTOR",
        "CORRECTOR_C1",
        "CORRECTOR_C2",
        "VERIFIER",
        "ARCHIVIST_A1",
        "ARCHIVIST_A2",
        "VALIDATOR",
    }
)


def rollback_state_to_node(state: dict, target_node: str) -> dict:
    """Deterministic state cleanup for safe revert to *target_node*.

    Clears only data that would cause the target node to consume
    stale downstream outputs.  Immutable core fields (``task``,
    ``task_id``) and ``audit_log`` are always preserved.

    Uses real flat ``GraphState`` fields only — never fabricates
    synthetic nested fields.

    Args:
        state: Current flat ``GraphState`` dict.
        target_node: Node to revert execution to.

    Returns:
        New state dict with downstream artifacts cleared.

    Raises:
        ValueError: If *target_node* is not a supported rollback target.
    """
    if target_node not in _SUPPORTED_ROLLBACK_TARGETS:
        raise ValueError(f"Unsupported rollback target: {target_node}")

    new_state = deepcopy(state)

    # Never modify immutable core:
    # new_state['task'], new_state['task_id'] preserved.
    # new_state['audit_log'] always preserved.

    if target_node == "ARCHITECT":
        # Full rollback: clear everything downstream of ARCHITECT.
        new_state["plan"] = None
        new_state["code"] = {}
        new_state["code_version"] = 0
        new_state["review_feedback"] = None
        new_state["test_results"] = None
        new_state["verifier_feedback"] = None
        new_state["applied_fixes"] = []
        new_state["status"] = None
        new_state["entry_status"] = None
        new_state["correction_attempt"] = 0
        new_state["verifier_reset_used"] = False
        logger.info("Rollback to ARCHITECT: all downstream state cleared")

    elif target_node == "CODER":
        # Clear everything downstream of CODER (preserving plan).
        new_state["code"] = {}
        new_state["code_version"] = 0
        new_state["review_feedback"] = None
        new_state["test_results"] = None
        new_state["verifier_feedback"] = None
        new_state["applied_fixes"] = []
        new_state["status"] = "plan_ready" if new_state.get("plan") else None
        new_state["entry_status"] = None
        new_state["correction_attempt"] = 0
        new_state["verifier_reset_used"] = False
        logger.info("Rollback to CODER: code and downstream state cleared")

    elif target_node == "REVIEWER":
        # Preserve code; clear review outputs and downstream.
        new_state["review_feedback"] = None
        new_state["test_results"] = None
        new_state["verifier_feedback"] = None
        new_state["status"] = "generated" if new_state.get("code") else None
        logger.info("Rollback to REVIEWER: review and downstream state cleared")

    elif target_node == "TESTER":
        new_state["test_results"] = None
        new_state["verifier_feedback"] = None
        new_state["applied_fixes"] = []
        new_state["correction_attempt"] = 0
        new_state["verifier_reset_used"] = False
        new_state["status"] = "OK"
        logger.info("Rollback to TESTER: test results, fixes, and counters cleared")

    elif target_node.startswith("CORRECTOR"):
        new_state["test_results"] = None
        new_state["verifier_feedback"] = None
        new_state["correction_attempt"] = 0
        new_state["verifier_reset_used"] = False
        new_state["status"] = "FAIL"
        logger.info(
            "Rollback to %s: test_results cleared, counters reset",
            target_node,
        )

    elif target_node == "VERIFIER":
        # Keep correction artifacts; clear verifier output.
        new_state["verifier_feedback"] = None
        logger.info("Rollback to VERIFIER: verifier_feedback cleared")

    elif target_node in ("ARCHIVIST_A1", "ARCHIVIST_A2", "VALIDATOR"):
        # Keep entry_status immutable for validator routing.
        # No state fields to clear — archival/validation nodes
        # operate on their own outputs.
        logger.info(
            "Rollback to %s: entry_status preserved, no downstream cleanup needed",
            target_node,
        )

    return new_state
