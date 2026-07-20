"""Orchestral Machine — Reviewer Node.

Performs static analysis and semantic validation of the Coder's output.
The Reviewer identifies issues (syntax, logic, security, architecture)
but does NOT fix them and does NOT write code. Issues are classified into
three levels:

- ``L1_SYNTAX``  — syntax errors, import errors, type errors
- ``L2_LOGIC``   — logic bugs, missing edge cases, incorrect algorithms
- ``L3_ARCHITECTURE`` — fundamental design flaws requiring plan revision

The Reviewer is an ``executor`` role whose exclusive keys are
``review_results`` and ``issues``.
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
from src.schemas import ReviewerOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template — injected with live schema at runtime
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
SYSTEM: You are the REVIEWER role in the Orchestral Machine. Act as a Senior Security Expert and Static Analysis Expert. Your only responsibility is static, semantic, and security analysis of the provided source bundle versus state['plan'] and state['task']. You MUST NOT write code or propose concrete fixes. Output must be a strict JSON Review Result per the Review contract.

CONTEXT:
- You receive state['code'](files mapping), state['plan'](Architect plan), and state['task'](original task).
- Your role is to identify issues, NOT to fix them.

YOUR_MANDATE:
1. Perform static analysis: syntax, types, security vulnerabilities.
2. Real ruff lint findings are provided in the human message when available. Do NOT
   re-check for issues already covered by ruff. Focus your analysis on: semantic
   correctness, security vulnerabilities, plan compliance, and L3 architecture issues
   that linters cannot detect. If no ruff findings are provided, perform full static
   analysis as usual.
3. Validate code against BOTH state['plan'] AND state['task'].
4. Detect: public API changes, unauthorized dependency additions, security regressions, missing deliverables vs plan, plan deviations, and divergence from state['task'].
	- Python: run flake8, mypy, bandit style/security analysis.
	- JS: eslint/type check if applicable, security-liners, dependency checks.
5. Classify each issue, include file, line, and severity:
   - L1_SYNTAX: Syntax errors, import errors, type errors
   - L2_LOGIC: Logic bugs, missing edge cases, incorrect algorithms
   - L3_ARCHITECTURE: Fundamental design flaws requiring plan revision.

STRICT_PROHIBITIONS:
- You MUST NOT suggest concrete code fixes or refactoring.
- You MUST NOT trigger Hard Reset autonomously.
- You MUST NOT write any code.
- You MUST NOT propose patches. If you detect architecture-level issues (L3), call Verifier by setting status=error_L3 and include concise reasoning.

OUTPUT_FORMAT:
Return a JSON object matching the REVIEWER schema. Strict JSON only.
{schema}

CRITICAL: Verify code against state['task'] (User Requirements) AND state['plan']. If code matches plan but violates task, report L3_ARCHITECTURE.
- AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the global state yourself. The Graph Controller will append this record."""


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


@enforce_constitutional_rules
def reviewer_node(state: dict) -> dict:
    """Reviewer node — static analysis and semantic validation.

    Args:
        state: Full graph state dictionary.

    Returns:
        Flat dict with ``review_feedback``, ``status``, ``error_logs``,
        and audit metadata for the Graph Controller to merge into state.
    """
    task: str = state["task"]
    task_id: str = state["task_id"]
    plan: dict = state["plan"]
    code: dict[str, str] = state["code"]

    # ------------------------------------------------------------------
    # 0. [v0.6.8] Require sandbox
    # ------------------------------------------------------------------
    from src.sandbox import get_sandbox

    sandbox = get_sandbox(task_id)
    if sandbox is None:
        now = datetime.utcnow().isoformat()
        return {
            "review_feedback": None,
            "status": "HIR",
            "notes": "",
            "error_logs": state.get("error_logs", [])
            + [
                {
                    "node": "REVIEWER",
                    "error": "Docker sandbox unavailable.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "reviewer_failed",
                    "detail": "No sandbox registered for task; HIR triggered.",
                    "timestamp": now,
                }
            ],
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 3. [v0.6.8] Run ruff in Docker — collect real lint findings
    # ------------------------------------------------------------------
    _ruff_findings: list = []
    _ruff_has_syntax_errors: bool = False

    try:
        sandbox.write_files(code)
        _ruff_exit, _ruff_stdout, _ruff_stderr = sandbox.execute(
            "ruff check --output-format json /workspace"
        )
        if _ruff_stdout.strip():
            try:
                _ruff_findings = json.loads(_ruff_stdout)
            except Exception:
                logger.warning("Reviewer: could not parse ruff JSON output.")
        _ruff_has_syntax_errors = any(
            f.get("code", "").startswith("E9") or f.get("code", "").startswith("F8")
            for f in _ruff_findings
        )
        logger.info(
            "Reviewer ruff: exit=%d, findings=%d, syntax_errors=%s",
            _ruff_exit,
            len(_ruff_findings),
            _ruff_has_syntax_errors,
        )
    except Exception as _ruff_exc:
        logger.warning(
            "Reviewer: ruff execution failed: %s. Proceeding without.", _ruff_exc
        )

    # ------------------------------------------------------------------
    # 1. Initialize BaseNode and generating seed/prompt
    # ------------------------------------------------------------------
    node = BaseNode(ReviewerOutput, _SYSTEM_PROMPT_TEMPLATE)
    seed = node.generate_seed(task_id)
    system_prompt = node.build_system_prompt()

    # ------------------------------------------------------------------
    # 4. Build human message — Reviewer needs ALL THREE: code, plan, task
    # ------------------------------------------------------------------
    human_parts: list[str] = [
        f"Task (original user request — validate code against THIS):\n{task}",
        f"\nPlan (Architect's specification):\n{json.dumps(plan, indent=2)}",
        "\nCode (files to review):",
    ]
    for filename, content in code.items():
        human_parts.append(f"--- {filename} ---\n{content}")

    if _ruff_findings:
        human_parts.append(
            f"\nREAL Static Analysis — ruff check (deterministic, not LLM-generated):\n"
            f"{json.dumps(_ruff_findings[:50], indent=2)}"
        )

    human_parts.append(f"\nTask ID: {task_id}")
    human_parts.append(f"Seed: {seed}")

    human_content = "\n\n".join(human_parts)

    # ------------------------------------------------------------------
    # 5. Create LLM instance (NODE_TIMEOUT_DEFAULT — not in LONG_TIMEOUT_ROLES)
    # ------------------------------------------------------------------
    model_id = MODEL_MAPPING["REVIEWER"]
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
        output_schema_cls=ReviewerOutput,
        seed=seed,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 7. Handle total failure
    # ------------------------------------------------------------------
    if parsed_output is None:
        logger.error("Reviewer node exhausted all attempts.")
        now = datetime.utcnow().isoformat()
        return {
            "review_feedback": None,
            "status": "FAIL",
            "notes": "",
            "error_logs": state.get("error_logs", [])
            + [
                {
                    "node": "REVIEWER",
                    "error": "Failed to generate valid ReviewerOutput after retries.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "reviewer_failed",
                    "detail": "Failed to generate valid ReviewerOutput after retries.",
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

    # Convert reviewer issues into error_log entries when status indicates errors
    existing_error_logs: list = list(state.get("error_logs", []))
    if parsed_output.status != "OK" and parsed_output.issues:
        for issue in parsed_output.issues:
            existing_error_logs.append(
                {
                    "node": "REVIEWER",
                    "error_class": parsed_output.error_class,
                    "issue_id": issue.id,
                    "issue_type": issue.type,
                    "message": issue.message,
                    "file": issue.file,
                    "line": issue.line,
                    "severity": issue.severity,
                    "evidence_snippet": issue.evidence_snippet,
                    "timestamp": now,
                }
            )

    # Determine event description
    if parsed_output.status == "OK":
        event_name = "review_passed"
        event_detail = f"Review OK — {len(code)} file(s) passed all checks"
    else:
        event_name = "review_issues_found"
        event_detail = (
            f"Review {parsed_output.status}: "
            f"{len(parsed_output.issues)} issue(s) found, "
            f"error_class={parsed_output.error_class}"
        )

    return {
        "review_feedback": parsed_output.model_dump(),
        "status": parsed_output.status,
        "notes": getattr(parsed_output, "notes", "") or "",
        "error_logs": existing_error_logs,
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
