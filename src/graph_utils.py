"""Orchestral Machine — Graph Utility Functions.

Shared helpers used by the Graph Controller, routing, and system nodes.
"""

import hashlib
from datetime import datetime
from typing import Any, Dict

from src.state import AuditEntry


def _now() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat()


def _seed(task_id: str) -> str:
    """Deterministic seed derived from task_id."""
    return hashlib.sha256(task_id.encode()).hexdigest()[:16]


def _ensure_dict(state) -> dict:
    """Convert state to dict if it's a Pydantic BaseModel.

    LangGraph may pass a Pydantic object when StateGraph is parameterized
    with a Pydantic model class. All node and routing functions expect dict.
    """
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return dict(state)


def _make_audit_entry(
    node_name: str,
    state: dict,
    node_output: dict,
    seed: str,
) -> AuditEntry:
    """Map node AuditMetadata output to canonical AuditEntry."""
    audit_meta = node_output.get("audit", {})
    if isinstance(audit_meta, dict):
        model_id = audit_meta.get("model_id", "unknown")
        cost_estimate = audit_meta.get("cost_estimate")
        prompt_summary = audit_meta.get("prompt_summary")
    else:
        model_id = "unknown"
        cost_estimate = None
        prompt_summary = None

    return AuditEntry(
        node=node_name,
        model_id=model_id,
        timestamp=_now(),
        status=node_output.get("status", "unknown"),
        prompt_summary=prompt_summary,
        input_hash=hashlib.sha256(str(state).encode()).hexdigest()[:16],
        output_hash=hashlib.sha256(str(node_output).encode()).hexdigest()[:16],
        cost_estimate=cost_estimate,
        seed=seed,
    )


def _make_system_audit_entry(
    status: str,
    prompt_summary: str,
    task_id: str,
) -> Dict[str, Any]:
    """Create a standardized system audit log entry."""
    return AuditEntry(
        node="GRAPH_CONTROLLER",
        model_id="system",
        timestamp=_now(),
        status=status,
        prompt_summary=prompt_summary,
        seed=_seed(task_id),
    ).model_dump()
