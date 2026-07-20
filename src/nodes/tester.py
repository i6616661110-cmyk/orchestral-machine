"""Orchestral Machine — Tester Node.

Generates test suites and executes them in an isolated Docker sandbox.
The Tester is an independent QA role: it generates its own tests based on the
task, plan, and code, writes them into the sandbox, runs pytest, and parses
the real output.

If the Docker sandbox is unavailable, the Tester returns HIR immediately.
The Tester does NOT modify application code to make tests pass.

The Tester is an ``executor`` role whose exclusive key is ``test_results``.
"""

import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import MODEL_MAPPING, NODE_TIMEOUT_DEFAULT
from src.enforcement import enforce_constitutional_rules
from src.llm_factory import build_chat_llm, invoke_node_llm
from src.nodes.base import BaseNode
from src.schemas import TesterOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal schema for test generation step (not exported)
# ---------------------------------------------------------------------------


class _TestGenOutput(BaseModel):
    """LLM output for the test generation step."""

    test_files: dict[str, str] = Field(
        description="Map of test file paths to their Python source content"
    )


# ---------------------------------------------------------------------------
# Prompt: test generation (LLM call #1)
# ---------------------------------------------------------------------------

_TEST_GEN_PROMPT_TEMPLATE = """\
SYSTEM: You are the TESTER role in the Orchestral Machine acting as QA Automation \
Lead. Your task is to generate pytest test files that verify the provided code \
against the task requirements and the Architect's plan.

CONTEXT:
- You receive the original task, the Architect's plan, and all code files from the Coder.
- The tests will run inside a Docker container with Python 3.12, pytest, pydantic, \
requests, and httpx.
- The code files are already written to /workspace/ in the container.

YOUR_MANDATE:
1. Generate one or more pytest test files that thoroughly verify the code.
2. Tests MUST import from the actual code files provided (use the exact module paths \
as they appear in the file listing).
3. Cover these categories:
   - Smoke test: Can the main module be imported without errors?
   - Functional tests: Does the code produce correct output for typical inputs?
   - Edge cases: Empty inputs, boundary values, invalid data handling.
   - Plan compliance: Does the code implement what the plan specifies?
   - Security: Verify that no API keys, passwords, or secrets are hardcoded in source \
files. Check that no .env literals appear in code.
   - Dependency check: Verify all imports resolve. If a module is imported but not in \
standard library or requirements, write a test that exposes the ImportError.
   - Idempotency: Where applicable, call the main logic twice with identical input and \
assert outputs are identical.
4. Each test function must have a clear, descriptive name starting with test_.
5. Use only pytest and standard library. Do NOT require packages not available in \
the sandbox.
6. Tests must be deterministic — no randomness, no network calls, no filesystem \
side effects outside /workspace.

STRICT_PROHIBITIONS:
- Do NOT generate tests that always pass regardless of code correctness.
- Do NOT mock the code under test — test real behavior.
- Do NOT generate tests for functionality not present in the code or plan.

OUTPUT_FORMAT:
Return a JSON object matching this schema. Strict JSON only.
{schema}

Use paths like "tests/test_smoke.py", "tests/test_functional.py", etc."""


# ---------------------------------------------------------------------------
# Prompt: output parsing (LLM call #2)
# ---------------------------------------------------------------------------

_PARSE_PROMPT_TEMPLATE = """\
SYSTEM: You are the TESTER role in the Orchestral Machine. You are a test output \
parser. You receive real pytest output captured from a Docker sandbox and must parse \
it into a structured JSON report.

CONTEXT:
- You receive actual stdout/stderr from running pytest inside a Docker container.
- The exit code tells you whether tests passed (0) or failed (non-zero).
- Parse suites, failures, and metadata from the actual output only.

YOUR_MANDATE:
1. Parse the test runner output into the TesterOutput schema exactly.
2. Identify each test suite name, its pass/fail/skip counts, and duration.
3. Extract individual failure details: test name, file, error type, traceback, message.
4. Set status to "PASS" if exit_code is 0, "FAIL" if non-zero.
5. Record environment.runner as "pytest".

STRICT_PROHIBITIONS:
- Do NOT invent test results not present in the output.
- Do NOT modify or comment on application code.
- Do NOT report PASS if the exit code is non-zero.

OUTPUT_FORMAT:
Return a JSON object matching the TESTER schema. Strict JSON only.
{schema}

AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the \
global state yourself. The Graph Controller will append this record."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hir_return(state: dict, error: str, detail: str) -> dict:
    now = datetime.utcnow().isoformat()
    return {
        "test_results": None,
        "status": "HIR",
        "error_logs": state.get("error_logs", [])
        + [{"node": "TESTER", "error": error, "timestamp": now}],
        "event_log": state.get("event_log", [])
        + [{"event": "tester_failed", "detail": detail, "timestamp": now}],
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


@enforce_constitutional_rules
def tester_node(state: dict) -> dict:
    """Tester node — generates tests, executes in Docker, parses output.

    Two LLM calls:
    1. Generate test files from task + plan + code.
    2. Parse real pytest output into TesterOutput.

    Args:
        state: Full graph state dictionary.

    Returns:
        Flat dict with ``test_results``, ``status``, ``error_logs``,
        and audit metadata for the Graph Controller to merge into state.
    """
    task: str = state["task"]
    task_id: str = state["task_id"]
    plan: dict = state.get("plan", {}) or {}
    code: dict[str, str] = state["code"]

    # ------------------------------------------------------------------
    # 1. Require sandbox
    # ------------------------------------------------------------------
    from src.sandbox import get_sandbox

    sandbox = get_sandbox(task_id)
    if sandbox is None:
        return _hir_return(
            state,
            error="Docker sandbox unavailable. Cannot execute tests.",
            detail="No sandbox registered for task; HIR triggered.",
        )

    # ------------------------------------------------------------------
    # 2. Write code files into sandbox
    # ------------------------------------------------------------------
    try:
        sandbox.write_files(code)
    except Exception as exc:
        return _hir_return(
            state,
            error=f"Failed to write code to sandbox: {exc}",
            detail="sandbox.write_files raised an exception.",
        )

    # ------------------------------------------------------------------
    # 3. Generate test files via LLM (call #1)
    # ------------------------------------------------------------------
    model_id = MODEL_MAPPING["TESTER"]
    gen_node = BaseNode(_TestGenOutput, _TEST_GEN_PROMPT_TEMPLATE)
    seed = gen_node.generate_seed(task_id)
    gen_system_prompt = gen_node.build_system_prompt()

    code_listing = "\n".join(
        f"--- {path} ---\n{content}" for path, content in code.items()
    )
    gen_human_content = "\n".join(
        [
            f"Task: {task}",
            f"\nPlan:\n{json.dumps(plan, indent=2)}",
            f"\nCode files:\n{code_listing}",
            f"\nTask ID: {task_id}",
            f"Seed: {seed}",
        ]
    )

    llm = build_chat_llm(
        model_id=model_id,
        timeout=NODE_TIMEOUT_DEFAULT,
        seed=int(seed, 16) % (2**31),
    )

    gen_output = invoke_node_llm(
        llm=llm,
        model_id=model_id,
        messages=[
            SystemMessage(content=gen_system_prompt),
            HumanMessage(content=gen_human_content),
        ],
        output_schema_cls=_TestGenOutput,
        seed=seed,
        logger=logger,
    )

    if gen_output is None:
        logger.error("Tester failed to generate test files.")
        return _hir_return(
            state,
            error="Test generation failure (all LLM attempts exhausted).",
            detail="Failed to generate test files after retries.",
        )

    if not gen_output.test_files:
        return _hir_return(
            state,
            error="LLM returned empty test_files dict.",
            detail="Test generation produced no test files.",
        )

    # ------------------------------------------------------------------
    # 4. Write generated test files to sandbox
    # ------------------------------------------------------------------
    try:
        sandbox.write_files(gen_output.test_files)
    except Exception as exc:
        return _hir_return(
            state,
            error=f"Failed to write test files to sandbox: {exc}",
            detail="sandbox.write_files raised an exception for generated tests.",
        )

    # ------------------------------------------------------------------
    # 5. Install dependencies if requirements.txt is present (non-fatal)
    # ------------------------------------------------------------------
    if "requirements.txt" in code:
        try:
            sandbox.execute("pip install -q -r requirements.txt")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 6. Run pytest
    # ------------------------------------------------------------------
    test_cmd = "python -m pytest -v -p no:cacheprovider"
    try:
        exit_code, stdout, stderr = sandbox.execute(test_cmd)
    except Exception as exc:
        return _hir_return(
            state,
            error=f"Test execution failed in sandbox: {exc}",
            detail="sandbox.execute raised an exception during pytest run.",
        )

    # ------------------------------------------------------------------
    # 7. Parse real output with LLM (call #2)
    # ------------------------------------------------------------------
    parse_node = BaseNode(TesterOutput, _PARSE_PROMPT_TEMPLATE)
    parse_system_prompt = parse_node.build_system_prompt()

    parse_human_content = "\n".join(
        [
            f"Task: {task}",
            f"Command: {test_cmd}",
            f"Exit code: {exit_code}",
            f"\nSTDOUT:\n{stdout or '(empty)'}",
            f"\nSTDERR:\n{stderr or '(empty)'}",
            f"\nTask ID: {task_id}",
            f"Seed: {seed}",
        ]
    )

    parsed_output = invoke_node_llm(
        llm=llm,
        model_id=model_id,
        messages=[
            SystemMessage(content=parse_system_prompt),
            HumanMessage(content=parse_human_content),
        ],
        output_schema_cls=TesterOutput,
        seed=seed,
        logger=logger,
    )

    if parsed_output is None:
        logger.error("Tester node exhausted all LLM parse attempts.")
        return _hir_return(
            state,
            error="Tester output parse failure (all LLM attempts exhausted).",
            detail="Failed to parse TesterOutput from real test output after retries.",
        )

    # ------------------------------------------------------------------
    # 8. Build successful state update
    # ------------------------------------------------------------------
    now = datetime.utcnow().isoformat()

    existing_error_logs: list = list(state.get("error_logs", []))
    if parsed_output.status == "FAIL" and parsed_output.failures:
        for failure in parsed_output.failures:
            existing_error_logs.append(
                {
                    "node": "TESTER",
                    "test": failure.test,
                    "file": failure.file,
                    "error_type": failure.error_type,
                    "failure_class": failure.type,
                    "message": failure.message,
                    "traceback": failure.traceback,
                    "timestamp": now,
                }
            )

    total_passed = sum(s.passed for s in parsed_output.suites)
    total_failed = sum(s.failed for s in parsed_output.suites)
    total_skipped = sum(s.skipped for s in parsed_output.suites)

    if parsed_output.status == "PASS":
        event_name = "tests_passed"
        event_detail = (
            f"All tests passed: {total_passed} passed, "
            f"{total_skipped} skipped across {len(parsed_output.suites)} suite(s)"
        )
    else:
        event_name = "tests_failed"
        event_detail = (
            f"Test failures: {total_failed} failed, {total_passed} passed "
            f"across {len(parsed_output.suites)} suite(s), "
            f"{len(parsed_output.failures)} failure detail(s)"
        )

    return {
        "test_results": parsed_output.model_dump(),
        "status": parsed_output.status,
        "error_logs": existing_error_logs,
        "event_log": state.get("event_log", [])
        + [{"event": event_name, "detail": event_detail, "timestamp": now}],
        "active_model": model_id,
        "updated_at": now,
    }
