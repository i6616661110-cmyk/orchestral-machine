"""Orchestral Machine — Data Access Control.

Defines which state fields each role may read/write and provides
filtering functions used by the Graph Controller.
"""

from copy import deepcopy
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

DATA_ACCESS_MATRIX: Dict[str, Dict[str, List[str]]] = {
    "ARCHITECT": {
        "read": [
            "task",
            "archived_meta_summary_ref",
            "archived_summary_ref",
            "code",
        ],
        "write": ["plan", "execution_strategy"],
    },
    "CODER": {
        "read": ["task", "plan", "error_logs", "archived_meta_summary_ref"],
        "write": ["code", "code_version"],
    },
    "REVIEWER": {
        "read": ["task", "plan", "code"],
        "write": ["review_feedback", "issues"],
    },
    "TESTER": {
        "read": ["task", "code", "plan"],
        "write": ["test_results"],
    },
    "CORRECTOR": {
        "read": [
            "code",
            "error_logs",
            "test_results",
            "verifier_feedback",
            "archived_summary_ref",
            "correction_attempt",
        ],
        "write": ["code", "applied_fixes"],
    },
    "VERIFIER": {
        "read": ["plan", "code", "error_logs", "correction_attempt"],
        "write": ["verifier_feedback"],
    },
    "ARCHIVIST_A1": {
        "read": [
            "archivist_queue",
            "error_logs",
            "archived_summary_ref",
            "archivist_feedback",
            "archival_attempt",
        ],
        "write": ["archived_summary_ref"],
    },
    "ARCHIVIST_A2": {
        "read": [
            "archivist_queue",
            "error_logs",
            "archived_summary_ref",
            "archivist_feedback",
            "archival_attempt",
        ],
        "write": ["archived_meta_summary_ref", "meta_summary"],
    },
    "VALIDATOR": {
        "read": [
            "archived_summary_ref",
            "archived_meta_summary_ref",
            "entry_status",
        ],
        "write": ["approved_hard_reset", "approved_summary", "approved_meta"],
    },
}

# Implicit read access for ALL roles
_IMPLICIT_READ_FIELDS: List[str] = ["task_id", "audit_log"]

# Fields managed exclusively by the Graph Controller — nodes may NOT write these
_CONTROLLER_EXCLUSIVE_FIELDS: frozenset = frozenset(
    {
        "audit_log",
        "loop_iteration",
        "correction_attempt",
        "archival_attempt",
        "meta_attempt",
        "validation_attempt",
        "entry_status",
        "archivist_queue",
    }
)


def _filter_state_for_node(state: dict, node_name: str) -> dict:
    """Return a copy of state filtered to only the fields the node may read."""
    matrix_key = node_name
    if matrix_key not in DATA_ACCESS_MATRIX:
        # Fallback: try without suffix (e.g. CORRECTOR)
        base = node_name.split("_")[0]
        matrix_key = base if base in DATA_ACCESS_MATRIX else node_name

    allowed_keys = DATA_ACCESS_MATRIX.get(matrix_key, {}).get("read", [])
    all_allowed = set(allowed_keys) | set(_IMPLICIT_READ_FIELDS)

    filtered: Dict[str, Any] = {}
    for key in all_allowed:
        if key in state:
            filtered[key] = deepcopy(state[key])

    return filtered


def _strip_controller_exclusive_fields(node_output: dict) -> dict:
    """Remove controller-exclusive fields from node output to prevent leakage."""
    cleaned = {}
    for key, value in node_output.items():
        if key not in _CONTROLLER_EXCLUSIVE_FIELDS:
            cleaned[key] = value
        else:
            logger.warning(
                "Stripped controller-exclusive field '%s' from node output",
                key,
            )
    return cleaned
