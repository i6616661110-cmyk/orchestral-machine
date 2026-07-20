"""Orchestral Machine — Shared State Definition.

Defines the complete state schema that flows through every node in the
LangGraph state machine.  All global state lives here.

Classes:
    AuditEntry  — per-node execution audit record.
    GraphState  — the single shared state object for the entire graph.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """Granular audit record emitted by each node on every execution.

    Captures the model used, timing, decision classification, and
    optional input/output hashes for deterministic replay verification.
    """

    node: str
    model_id: str
    timestamp: str
    status: str
    prompt_summary: Optional[str] = None
    decision_type: Optional[str] = None
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    cost_estimate: Optional[float] = None
    seed: Optional[str] = None


class GraphState(BaseModel):
    """Complete state schema for Orchestral Machine.

    Organized into logical sections: immutable core identifiers,
    architectural work products, routing/status flags, counters,
    feedback from review/test/verify, logs & audit trail,
    archive references, and session metadata.
    """

    # ========== IMMUTABLE CORE ==========
    task: str  # Original user request — NEVER modified
    task_id: str  # Unique session ID

    # ========== ARCHITECTURAL ==========
    plan: Optional[Dict[str, Any]] = None  # Plan JSON from ARCHITECT
    execution_strategy: Optional[str] = None  # Architect's chosen execution approach
    code: Dict[str, str] = Field(default_factory=dict)  # filename → content
    code_version: Optional[int] = 0  # Track code generation versions

    # ========== ROUTING & STATUS ==========
    status: Optional[str] = None  # e.g. 'PASS', 'FAIL', 'HIR', 'running'
    entry_status: Optional[str] = (
        None  # Status before Validator/Archivist (resume point)
    )

    # ========== COUNTERS ==========
    loop_iteration: int = 0  # Full rewrites by Architect (>2 → HIR)
    correction_attempt: int = 0  # Coder/Corrector cycles
    archival_attempt: int = 0  # Archivist_A1 sessions
    meta_attempt: int = 0  # Archivist_A2 (Meta-summary) attempts
    validation_attempt: int = 0  # Validator attempts

    # ========== FLAGS & SYSTEM STATE ==========
    verifier_reset_used: bool = (
        False  # Verifier forced correction_attempt reset (once per loop_iteration)
    )
    system_flags: Dict[str, bool] = Field(default_factory=dict)
    resume_target: Optional[str] = None  # Target node for resume (skip nodes before it)
    # Expected keys:
    #   'halted'             — emergency stop requested
    #   'recursion_warning'  — approaching LangGraph recursion limit
    #   'checkpoint_saved'   — latest checkpoint persisted
    #   'fallback_active'    — RESERVIST model in use
    #   'state_size_warning' — state approaching size threshold

    # ========== FEEDBACK & WORK PRODUCTS ==========
    review_feedback: Optional[Dict[str, Any]] = None  # Reviewer output
    test_results: Optional[Dict[str, Any]] = None  # Tester output
    verifier_feedback: Optional[Dict[str, Any]] = None  # Verifier output
    applied_fixes: List[Dict[str, Any]] = Field(default_factory=list)  # Corrector fixes
    archivist_feedback: Optional[List[str]] = (
        None  # Validator/A2 instructions for A1 retries
    )
    archivist_queue: Optional[Dict[str, Any]] = (
        None  # Raw data queue for Archivist processing
    )

    # ========== LOGS & AUDIT ==========
    error_logs: List[Dict[str, Any]] = Field(default_factory=list)  # Last 20 errors
    event_log: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # High-level event log (UI/Debug)
    audit_log: List[AuditEntry] = Field(
        default_factory=list
    )  # Detailed node audit trail

    # ========== ARCHIVES & REFS ==========
    archived_summary_ref: Optional[Union[str, Dict[str, Any]]] = (
        None  # Qdrant ref (not payload)
    )
    meta_summary: Optional[Dict[str, Any]] = None  # Ephemeral summary for Architect
    archived_meta_summary_ref: Optional[Union[str, Dict[str, Any]]] = (
        None  # Validated meta-summary ref in Vector DB
    )
    archived_error_refs: List[str] = Field(
        default_factory=list
    )  # Error refs in vector DB
    snapshot_path: Optional[str] = None  # File system snapshot/backup path

    # ========== METADATA ==========
    notes: Optional[str] = None  # Node-to-human commentary for status updates
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    active_model: Optional[str] = None  # Model currently processing
