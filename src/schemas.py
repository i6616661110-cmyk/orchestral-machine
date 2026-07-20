"""Orchestral Machine — Strict JSON Output Contracts.

Every node in the graph MUST return JSON that validates against one of the
Pydantic models defined here.  The Graph Controller deserialises each node
response and runs ``Model.model_validate(payload)`` before applying any
state mutations.

Nodes inject the schema into their system prompts at runtime via:
    ``json.dumps(Model.model_json_schema(), indent=2)``

Classes exported (10 total):
    BaseNodeOutput          — common envelope inherited by all node outputs
    ArchitectOutput         — ARCHITECT plan
    CoderOutput             — CODER generated files
    ReviewerOutput          — REVIEWER analysis
    CorrectorOutput         — CORRECTOR fixes
    VerifierOutput          — VERIFIER meta-control decisions
    TesterOutput            — TESTER execution results
    ArchivistA1Output       — ARCHIVIST_A1 archival records
    ArchivistA2Output       — ARCHIVIST_A2 meta-summary
    ValidatorOutput         — VALIDATOR quality gate
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class AuditMetadata(BaseModel):
    """Per-output audit payload attached to every node response.

    Distinct from ``AuditEntry`` in ``state.py`` — this is the raw audit
    blob produced by the node itself, before the controller normalises it
    into the state-level audit log.
    """

    node: str
    model_id: str
    prompt_summary: Optional[str] = None
    timestamp: str
    cost_estimate: Optional[float] = 0.0


# ---------------------------------------------------------------------------
# Base output envelope
# ---------------------------------------------------------------------------

class BaseNodeOutput(BaseModel):
    """Common envelope inherited by every role-specific output schema."""

    node: str
    status: str
    events: List[str] = Field(default_factory=list)
    timestamp: str
    confidence: Optional[float] = None
    notes: Optional[str] = Field(
        default=None, 
        description="Always provide a brief summary of your work, fixes, or findings here."
    )
    audit: AuditMetadata


# ---------------------------------------------------------------------------
# ARCHITECT
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    """Single step inside an Architect plan."""

    id: str
    title: str
    desc: str


class PlanMetadata(BaseModel):
    """Metadata block for the Architect plan."""

    created_by: str
    timestamp: str


class ArchitectOutput(BaseNodeOutput):
    """ARCHITECT node output — implementation plan."""

    status: Literal["plan_ready"]
    task_id: str
    objective: str
    constraints: List[str] = Field(default_factory=list)
    deliverables: List[str] = Field(default_factory=list)
    steps: List[PlanStep] = Field(default_factory=list)
    version: int = 1
    metadata: PlanMetadata


# ---------------------------------------------------------------------------
# CODER
# ---------------------------------------------------------------------------

class CoderOutput(BaseNodeOutput):
    """CODER node output — generated source files."""

    status: Literal["generated"]
    files: Dict[str, str]
    entrypoint: str
    dependencies: List[str] = Field(default_factory=list)
    code_version: int = 1
    previous_errors_considered: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# REVIEWER
# ---------------------------------------------------------------------------

class ReviewIssue(BaseModel):
    """Single issue identified during code review."""

    id: str
    type: str
    message: str
    evidence_snippet: str
    file: str
    line: int
    severity: Literal["low", "high", "critical"]


class ReviewerOutput(BaseNodeOutput):
    """REVIEWER node output — static / semantic / security analysis."""

    status: Literal["OK", "error_L1", "error_L2", "error_L3"]
    error_class: Optional[Literal["L1_SYNTAX", "L2_LOGIC", "L3_ARCHITECTURE"]] = None
    summary: str
    tool_output: Optional[str] = None
    issues: List[ReviewIssue] = Field(default_factory=list)
    traceback: Optional[str] = None
    suggested_fix: Optional[str] = None


# ---------------------------------------------------------------------------
# CORRECTOR
# ---------------------------------------------------------------------------

class AppliedFix(BaseModel):
    """Single fix applied by the Corrector."""

    id: str
    desc: str
    diff: Optional[str] = None


class CorrectorOutput(BaseNodeOutput):
    """CORRECTOR node output — surgical code fixes."""

    status: Literal["fixed", "NEEDS_REWRITE", "no_change"]
    fixed_code: Dict[str, str] = Field(default_factory=dict)
    request_increment_correction_attempt: bool = True
    historical_context: Optional[str] = None
    applied_fixes: List[AppliedFix] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# VERIFIER
# ---------------------------------------------------------------------------

class CheckedInputs(BaseModel):
    """Snapshot of input versions the Verifier inspected."""

    plan_version: Optional[int] = None
    code_version: Optional[int] = None
    error_log_count: Optional[int] = None


class VerifierOutput(BaseNodeOutput):
    """VERIFIER node output — meta-control decisions."""

    status: Literal[
        "confirmed_L3",
        "disagree_L3",
        "confirmed_needs_rewrite",
        "execution_failure",
    ]
    reason: str
    reset_recommendation: bool = False
    checked_inputs: Optional[CheckedInputs] = None


# ---------------------------------------------------------------------------
# TESTER
# ---------------------------------------------------------------------------

class TestSuite(BaseModel):
    """Result summary for a single test suite."""

    suite_name: str
    tests_count: int
    passed: int
    failed: int
    skipped: int = 0
    duration_ms: Optional[int] = None


class TestFailure(BaseModel):
    """Detail record for a single test failure."""

    test: str
    file: str
    error_type: str
    traceback: Optional[str] = None
    type: Optional[Literal["LOGIC", "SYNTAX"]] = None
    message: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class CoverageReport(BaseModel):
    """Test coverage summary."""

    percent: float
    report_file: Optional[str] = None


class TestArtifacts(BaseModel):
    """Artifacts produced during the test run."""

    logs: Optional[str] = None
    screenshots: List[str] = Field(default_factory=list)
    coverage: Optional[CoverageReport] = None


class TestEnvironment(BaseModel):
    """Execution environment metadata."""

    runner: str
    python_version: Optional[str] = None
    node_version: Optional[str] = None
    container_id: Optional[str] = None


class TestDeterminism(BaseModel):
    """Determinism tracking for test runs."""

    seed_used: Optional[str] = None
    notes: Optional[str] = None


class TesterOutput(BaseNodeOutput):
    """TESTER node output — deterministic test execution results."""

    status: Literal["PASS", "FAIL"]
    tests_run: List[str] = Field(default_factory=list)
    suites: List[TestSuite] = Field(default_factory=list)
    failures: List[TestFailure] = Field(default_factory=list)
    artifacts: Optional[TestArtifacts] = None
    execution_time_ms: Optional[int] = None
    environment: Optional[TestEnvironment] = None
    determinism: Optional[TestDeterminism] = None


# ---------------------------------------------------------------------------
# ARCHIVIST_A1
# ---------------------------------------------------------------------------

class CodeContext(BaseModel):
    """Source-code context for an archival record."""

    file: str
    lines: str
    snippet: str


class ArchivalRecord(BaseModel):
    """Single record created during archival."""

    record_id: str
    collection: str
    error_type: Optional[str] = None
    facts: List[str] = Field(default_factory=list)
    hypotheses: List[str] = Field(default_factory=list)
    code_context: Optional[CodeContext] = None


class ArchivistA1Output(BaseNodeOutput):
    """ARCHIVIST_A1 node output — failure record archival."""

    status: Literal["archived", "archival_error"]
    increment_archival_attempt: bool = True
    records_created: List[ArchivalRecord] = Field(default_factory=list)
    embedding_model: Optional[str] = None


# ---------------------------------------------------------------------------
# ARCHIVIST_A2
# ---------------------------------------------------------------------------

class RootCause(BaseModel):
    """Identified root cause from meta-analysis."""

    cause_id: str
    description: str
    frequency: int
    severity: Literal["low", "medium", "high", "critical"]


class FailurePattern(BaseModel):
    """Recurring failure pattern detected across records."""

    pattern_id: str
    description: str
    occurrences: int


class MetaStatistics(BaseModel):
    """Aggregated error statistics from the meta-summary."""

    total_attempts: int
    l1_errors: int = 0
    l2_errors: int = 0
    l3_errors: int = 0


class ArchivistA2Output(BaseNodeOutput):
    """ARCHIVIST_A2 node output — meta-summary creation."""

    status: Literal[
        "archived_meta_summary",
        "meta_summary_error",
        "archived_summary_corrupt",
    ]
    correction_instructions: List[str] = Field(default_factory=list)
    meta_summary_id: Optional[str] = None
    source_records: List[str] = Field(default_factory=list)
    root_causes: List[RootCause] = Field(default_factory=list)
    failure_patterns: List[FailurePattern] = Field(default_factory=list)
    recommendations_for_architect: List[str] = Field(default_factory=list)
    statistics: Optional[MetaStatistics] = None


# ---------------------------------------------------------------------------
# VALIDATOR
# ---------------------------------------------------------------------------

class ValidatorOutput(BaseNodeOutput):
    """VALIDATOR node output — quality gate for archives and transitions."""

    status: Literal[
        "ready_for_next",
        "approved_hard_reset",
        "HIR",
        "COMPLETED",
        "unapproved_archived_summary",
        "unapproved_meta_summary",
    ]
    checked_refs: List[str] = Field(default_factory=list)
    reason: str
    correction_instructions: List[str] = Field(default_factory=list)
    intended_next_action: List[str] = Field(default_factory=list)
