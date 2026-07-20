"""Orchestral Machine -- Corrector Node.

Dual-mode surgical code corrector with internal escalation routing.
The Corrector receives error logs and applies minimal, targeted L1/L2 fixes
without modifying architecture or public interfaces.

Escalation logic:
    - ``correction_attempt < 3``  -> CORRECTOR_C1 (lighter model)
    - ``correction_attempt > 2``  -> CORRECTOR_C2 (stronger model)
    - ``correction_attempt > MAX_CORRECTION_ATTEMPTS``  -> immediate NEEDS_REWRITE (no LLM call)

The Corrector is an ``executor`` role whose exclusive key is
``applied_fixes``.  It MUST NOT mutate ``correction_attempt`` directly --
it requests the increment via ``request_increment_correction_attempt``.
"""

import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    MAX_CORRECTION_ATTEMPTS,
    MODEL_MAPPING,
    NODE_REGEN_MAX,
    NODE_TIMEOUT_DEFAULT,
)
from src.enforcement import enforce_constitutional_rules
from src.llm_factory import build_chat_llm, invoke_node_llm
from src.nodes.base import BaseNode
from src.schemas import CorrectorOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt templates -- injected with live schema at runtime
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_C1_TEMPLATE = """\
SYSTEM: You are CORRECTOR_C1 role in the Orchestral Machine. Act as a Surgical Code Corrector performing minimal, targeted fixes. Your task is to perform deep, localized, surgical fixes to the provided code bundle to address L1 (syntax) and L2 (logic) issues reported in state['error_logs'] while taking into account historical failure context. You MUST NOT change architecture or public interfaces.

Note: Do not mutate global counters. In your output JSON include "request_increment_correction_attempt": 1 or "request_set_needs_rewrite": true. The Graph Controller will apply the counter increment atomically.

CONTEXT:
- You receive state['code'](files mapping), state['error_logs'](issues to fix), state['correction_attempt'](current counter).
- You may receive state['test_results'] and state['verifier_feedback'].
- Optional archived_summary_ref (id or metadata) — consult validated archival summaries to avoid repeated errors.

YOUR_MANDATE:
1. Analyze previous correction attempts (if any) and archived summaries; produce a targeted fix plan and apply fixes.
2. Apply ONLY fixes that address reported issues in state['error_logs'] or state['test_results'].
3. Make minimal, surgical changes — no refactoring.
4. Preserve all code not related to reported issues.
   - If you cannot safely fix without rewriting or fix requires architectural change or broad rewrite, set status="NEEDS_REWRITE".
5. Generate unified diff for each fix when possible.
6. On success, include unified diffs for each modified file.

STRICT_PROHIBITIONS:
- You MUST NOT make architectural changes.
- You MUST NOT refactor unrelated code.
- You MUST NOT reinterpret requirements.
- You MUST NOT modify public interfaces unless explicitly instructed by Reviewer.
- No refactors outside affected lines.
- Preserve file encoding and headers.

OUTPUT_FORMAT:
Attempt minimal edit(s) and return a JSON object matching the Corrector schema. Strict JSON only.
{schema}

Include status, complete fixed_code per file, request increment correction_attempt and applied_fixes array with unified diffs.
- Do NOT mutate `correction_attempt` yourself.
- AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the global state yourself. The Graph Controller will append this record."""

_SYSTEM_PROMPT_C2_TEMPLATE = """\
SYSTEM: You are CORRECTOR_C2 role in the Orchestral Machine. Act as a Surgical Code Corrector performing minimal, targeted fixes. You are invoked when earlier corrective attempts failed. Your task is to perform deeper, localized, surgical fixes to the provided code bundle to address L1 (syntax) and L2 (logic) issues reported in state['error_logs'] while taking into account historical failure context. You MUST NOT change architecture or public interfaces.

CONTEXT:
- You receive state['code'](files mapping), state['error_logs'](issues to fix), state['correction_attempt'](current counter).
- You may receive state['test_results'] and state['verifier_feedback'].
- Optional archived_summary_ref (id or metadata) — consult validated archival summaries to avoid repeated errors.

YOUR_MANDATE:
1. Analyze previous correction attempts and archived summaries; produce a targeted fix plan and apply fixes.
2. Apply ONLY fixes that address reported issues.
3. Preserve file encoding and headers.
4. Make minimal, surgical changes — no refactoring.
5. Preserve all code not related to reported issues.
   - If you cannot safely fix without rewriting or fix requires architectural change or broad rewrite, set status="NEEDS_REWRITE".
6. Generate unified diff for each fix when possible.
7. On success, include unified diffs for each modified file.

STRICT_PROHIBITIONS:
- You MUST NOT make architectural changes.
- You MUST NOT refactor unrelated code.
- You MUST NOT reinterpret requirements.
- You MUST NOT modify public interfaces unless explicitly instructed by Reviewer.
- No refactors outside affected lines.

OUTPUT_FORMAT:
Attempt minimal edit(s) and return a JSON object matching the Corrector schema. Strict JSON only.
{schema}

Include status, complete fixed_code per file, request_increment_correction_attempt and applied_fixes array with unified diffs.
- AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the global state yourself. The Graph Controller will append this record.

ESCALATION_CONTEXT:
You are invoked after CORRECTOR_C1 failed to resolve issues (correction_attempt = 0, 1, 2).
Carefully review state['error_logs'] to understand why previous fixes failed.
Consider edge cases and deeper logic issues that C1 may have missed."""


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


@enforce_constitutional_rules
def corrector_node(state: dict) -> dict:
    """Corrector node -- dual-mode surgical code corrector.

    Routes to CORRECTOR_C1 or CORRECTOR_C2 based on ``correction_attempt``,
    or returns NEEDS_REWRITE immediately when attempts exceed threshold.

    Args:
        state: Full graph state dictionary.

    Returns:
        Flat dict with ``code``, ``applied_fixes``, ``status``, and audit
        metadata for the Graph Controller to merge into state.
    """
    code: dict[str, str] = state["code"]
    error_logs: list = state.get("error_logs", [])
    test_results: dict | None = state.get("test_results")
    verifier_feedback: dict | None = state.get("verifier_feedback")
    archived_summary_ref = state.get("archived_summary_ref")
    correction_attempt: int = state.get("correction_attempt", 0)
    task_id: str = state["task_id"]

    # ------------------------------------------------------------------
    # 0. [v0.6.7] Require sandbox
    # ------------------------------------------------------------------
    from src.sandbox import get_sandbox

    sandbox = get_sandbox(task_id)
    if sandbox is None:
        now = datetime.utcnow().isoformat()
        return {
            "code": code,
            "applied_fixes": [],
            "status": "HIR",
            "notes": "",
            "error_logs": error_logs
            + [
                {
                    "node": "CORRECTOR",
                    "error": "Docker sandbox unavailable.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "corrector_failed",
                    "detail": "No sandbox registered for task; HIR triggered.",
                    "timestamp": now,
                }
            ],
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 1. Logic Guard: Prevent hallucination on empty error logs
    # ------------------------------------------------------------------
    # If the previous node (Tester) failed technically (status="FAIL" or "HIR")
    # and produced no error logs, the Corrector has nothing to fix.
    if not error_logs and state.get("status") == "FAIL":
        now = datetime.utcnow().isoformat()
        logger.error(
            "Corrector invoked with FAIL status but empty error_logs. Preventing hallucination."
        )
        return {
            "code": code,
            "applied_fixes": [],
            "status": "HIR",
            "request_increment_correction_attempt": True,  # Increment to avoid infinite loop at same attempt
            "error_logs": [
                {
                    "node": "CORRECTOR",
                    "error": "Corrector invoked with FAIL status but empty error_logs. Preventing hallucination.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "corrector_guard_hit",
                    "detail": "Empty error_logs with FAIL status. logic guard triggered.",
                    "timestamp": now,
                }
            ],
            "active_model": "none",
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 2. Escalation: correction_attempt > MAX_CORRECTION_ATTEMPTS -> immediate NEEDS_REWRITE
    # ------------------------------------------------------------------
    if correction_attempt > MAX_CORRECTION_ATTEMPTS:
        now = datetime.utcnow().isoformat()
        logger.warning(
            "Corrector: correction_attempt=%d > %d, returning NEEDS_REWRITE",
            correction_attempt,
            MAX_CORRECTION_ATTEMPTS,
        )
        return {
            "code": code,
            "applied_fixes": [],
            "status": "NEEDS_REWRITE",
            "request_increment_correction_attempt": True,
            "error_logs": error_logs
            + [
                {
                    "node": "CORRECTOR",
                    "error": (
                        f"Correction attempt {correction_attempt} exceeds "
                        f"threshold (>{MAX_CORRECTION_ATTEMPTS}). Escalating to NEEDS_REWRITE."
                    ),
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "corrector_escalated_needs_rewrite",
                    "detail": (
                        f"correction_attempt={correction_attempt} > {MAX_CORRECTION_ATTEMPTS}, "
                        "no LLM call, immediate NEEDS_REWRITE"
                    ),
                    "timestamp": now,
                }
            ],
            "active_model": "none",
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 3. Select model and prompt template based on correction_attempt
    #    (with infrastructure fallback override)
    # ------------------------------------------------------------------
    fallback_active = bool(state.get("system_flags", {}).get("fallback_active"))
    if fallback_active:
        model_id = MODEL_MAPPING["CORRECTOR_C2"]
        system_prompt_template = _SYSTEM_PROMPT_C2_TEMPLATE
        corrector_level = "C2"
        logger.warning(
            "Infrastructure fallback active: forcing escalation to CORRECTOR_C2 "
            "(attempt=%d, model=%s)",
            correction_attempt,
            model_id,
        )
    elif correction_attempt < 3:
        model_id = MODEL_MAPPING["CORRECTOR_C1"]
        system_prompt_template = _SYSTEM_PROMPT_C1_TEMPLATE
        corrector_level = "C1"
    else:
        model_id = MODEL_MAPPING["CORRECTOR_C2"]
        system_prompt_template = _SYSTEM_PROMPT_C2_TEMPLATE
        corrector_level = "C2"

    logger.info(
        "Corrector routing: attempt=%d -> %s (model=%s)",
        correction_attempt,
        corrector_level,
        model_id,
    )

    # ------------------------------------------------------------------
    # 4. Initialize BaseNode and generating seed/prompt
    # ------------------------------------------------------------------
    node = BaseNode(CorrectorOutput, system_prompt_template)
    seed = node.generate_seed(task_id)
    system_prompt = node.build_system_prompt()

    # ------------------------------------------------------------------
    # 6. Build human message -- include ALL readable fields
    # ------------------------------------------------------------------
    human_parts: list[str] = [
        f"Error Logs (issues to fix):\n{json.dumps(error_logs, indent=2)}",
        "\nCode (files mapping):",
    ]
    for filename, content in code.items():
        human_parts.append(f"--- {filename} ---\n{content}")

    if test_results is not None:
        human_parts.append(f"\nTest Results:\n{json.dumps(test_results, indent=2)}")

    if verifier_feedback is not None:
        human_parts.append(
            f"\nVerifier Feedback:\n{json.dumps(verifier_feedback, indent=2)}"
        )

    if archived_summary_ref is not None:
        human_parts.append(
            f"\nArchived Summary Reference:\n{json.dumps(archived_summary_ref, indent=2) if isinstance(archived_summary_ref, dict) else str(archived_summary_ref)}"
        )

    human_parts.append(f"\nCorrection Attempt: {correction_attempt}")
    human_parts.append(f"Task ID: {task_id}")
    human_parts.append(f"Seed: {seed}")

    human_content = "\n\n".join(human_parts)

    # ------------------------------------------------------------------
    # 7. Create LLM instance (NODE_TIMEOUT_DEFAULT -- not in LONG_TIMEOUT_ROLES)
    # ------------------------------------------------------------------
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
    # 8. Invoke LLM with JSON parse retry logic
    # ------------------------------------------------------------------
    parsed_output = invoke_node_llm(
        llm=llm,
        model_id=model_id,
        messages=messages,
        output_schema_cls=CorrectorOutput,
        seed=seed,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 8b. [v0.6.7] Self-verification: generate test, run in sandbox
    #     Only when status == "fixed" and fixed_code is non-empty.
    #     LLM call #2: generate mini-test. LLM call #3: retry fix if test fails.
    # ------------------------------------------------------------------
    if (
        parsed_output is not None
        and parsed_output.status == "fixed"
        and parsed_output.fixed_code
    ):
        _merged_for_test = dict(code)
        _merged_for_test.update(parsed_output.fixed_code)
        sandbox.write_files(_merged_for_test)

        # --- Step 1: Generate a mini-test for the fix (one-time) ---
        _test_gen_prompt = (
            "You are a test engineer. Write a minimal pytest test that verifies "
            "the fix described below. Return ONLY a JSON object: "
            '{"test_code": "<valid Python pytest code>"}\n\n'
            f"Error logs (what was broken):\n{json.dumps(error_logs[-5:], indent=2)}\n\n"
            f"Applied fixes:\n{json.dumps([f.model_dump() for f in parsed_output.applied_fixes], indent=2)}\n\n"
            "Fixed code files:\n"
        )
        for _fn, _fc in parsed_output.fixed_code.items():
            _test_gen_prompt += f"--- {_fn} ---\n{_fc[:2000]}\n"

        _test_code = None
        try:
            _test_gen_result = llm.invoke(
                [
                    SystemMessage(content=_test_gen_prompt),
                    HumanMessage(content="Generate a pytest test file."),
                ]
            )
            _test_gen_content = (
                _test_gen_result.content
                if hasattr(_test_gen_result, "content")
                else str(_test_gen_result)
            )
            _test_gen_text = str(_test_gen_content)
            _test_gen_json = json.loads(
                _test_gen_text[
                    _test_gen_text.index("{") : _test_gen_text.rindex("}") + 1
                ]
            )
            _test_code = _test_gen_json.get("test_code", "")
        except Exception as _tg_exc:
            logger.warning(
                "Corrector test generation failed: %s. Skipping self-verification.",
                _tg_exc,
            )

        # --- Step 2: Self-test retry loop (up to 3 attempts) ---
        _MAX_SELF_TEST_RETRIES = 3

        if _test_code and _test_code.strip():
            sandbox.write_files({"test_corrector_verify.py": _test_code})
            _self_test_passed = False

            for _retry_i in range(_MAX_SELF_TEST_RETRIES):
                _current_merged = dict(code)
                _current_merged.update(parsed_output.fixed_code)
                sandbox.write_files(_current_merged)

                try:
                    _retest_exit, _retest_stdout, _retest_stderr = sandbox.execute(
                        "python -m pytest /workspace/test_corrector_verify.py -v --tb=short"
                    )
                except Exception as _retest_exc:
                    logger.warning(
                        "Corrector self-test sandbox error: %s. Skipping.", _retest_exc
                    )
                    break

                if _retest_exit == 0:
                    _self_test_passed = True
                    logger.info(
                        "Corrector self-test PASSED (attempt %d/%d).",
                        _retry_i + 1,
                        _MAX_SELF_TEST_RETRIES,
                    )
                    break

                logger.info(
                    "Corrector self-test FAILED (exit=%d, attempt %d/%d).",
                    _retest_exit,
                    _retry_i + 1,
                    _MAX_SELF_TEST_RETRIES,
                )

                if _retry_i + 1 >= _MAX_SELF_TEST_RETRIES:
                    logger.warning(
                        "Corrector exhausted %d self-test retries. Passing to Reviewer.",
                        _MAX_SELF_TEST_RETRIES,
                    )
                    break

                # LLM retry with real test output
                _retry_human_parts = [
                    f"Error Logs (original issues):\n{json.dumps(error_logs, indent=2)}",
                    "\nYour fix was applied but a verification test still fails.",
                    f"\nTest code:\n{_test_code}",
                    f"\nReal pytest output (exit_code={_retest_exit}):",
                    f"--- stdout ---\n{_retest_stdout[:3000]}",
                    f"--- stderr ---\n{_retest_stderr[:1000]}",
                    f"\nThis is retry {_retry_i + 2}/{_MAX_SELF_TEST_RETRIES}. "
                    "Apply additional surgical fixes. Return complete CorrectorOutput.",
                    f"\nTask ID: {task_id}",
                    f"Seed: {seed}",
                ]
                _retry_output = invoke_node_llm(
                    llm=llm,
                    model_id=model_id,
                    messages=[
                        SystemMessage(content=system_prompt),
                        HumanMessage(content="\n\n".join(_retry_human_parts)),
                    ],
                    output_schema_cls=CorrectorOutput,
                    seed=seed,
                    logger=logger,
                )
                if _retry_output is not None and _retry_output.fixed_code:
                    parsed_output = _retry_output
                    logger.info(
                        "Corrector retry %d produced new fix (status=%s). Re-testing.",
                        _retry_i + 1,
                        parsed_output.status,
                    )
                else:
                    logger.warning(
                        "Corrector retry %d produced no usable output. Stopping retries.",
                        _retry_i + 1,
                    )
                    break

            if not _self_test_passed:
                logger.warning(
                    "Corrector self-test never passed after %d attempts. "
                    "Forwarding to Reviewer as-is.",
                    _MAX_SELF_TEST_RETRIES,
                )
        else:
            logger.info(
                "Corrector test generation returned empty. Skipping self-verification."
            )

    # ------------------------------------------------------------------
    # 9. Handle total failure
    # ------------------------------------------------------------------
    if parsed_output is None:
        logger.error(
            "Corrector node (%s) exhausted all attempts.",
            corrector_level,
        )
        now = datetime.utcnow().isoformat()
        return {
            "code": code,
            "applied_fixes": [],
            "status": "FAIL",
            "error_logs": error_logs
            + [
                {
                    "node": f"CORRECTOR_{corrector_level}",
                    "error": "Failed to generate valid CorrectorOutput after retries.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "corrector_failed",
                    "detail": "Failed to generate valid CorrectorOutput after retries.",
                    "timestamp": now,
                }
            ],
            "active_model": model_id,
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 10. Build successful state update
    # ------------------------------------------------------------------
    now = datetime.utcnow().isoformat()

    # Merge fixed_code into existing code (only replace modified files)
    merged_code = dict(code)
    if parsed_output.fixed_code:
        merged_code.update(parsed_output.fixed_code)

    # Append error log entries when status is NEEDS_REWRITE
    existing_error_logs: list = list(error_logs)
    if parsed_output.status == "NEEDS_REWRITE":
        existing_error_logs.append(
            {
                "node": f"CORRECTOR_{corrector_level}",
                "error": (
                    f"Corrector {corrector_level} determined NEEDS_REWRITE "
                    f"at attempt {correction_attempt}"
                ),
                "timestamp": now,
            }
        )

    # Determine event description
    if parsed_output.status == "fixed":
        event_name = "corrector_fixed"
        event_detail = (
            f"Corrector {corrector_level} applied "
            f"{len(parsed_output.applied_fixes)} fix(es) to "
            f"{len(parsed_output.fixed_code)} file(s)"
        )
    elif parsed_output.status == "NEEDS_REWRITE":
        event_name = "corrector_needs_rewrite"
        event_detail = (
            f"Corrector {corrector_level} escalated to NEEDS_REWRITE "
            f"at attempt {correction_attempt}"
        )
    else:
        event_name = "corrector_no_change"
        event_detail = (
            f"Corrector {corrector_level} found no changes needed "
            f"at attempt {correction_attempt}"
        )

    return {
        "code": merged_code,
        "applied_fixes": [fix.model_dump() for fix in parsed_output.applied_fixes],
        "status": parsed_output.status,
        "notes": parsed_output.notes,
        "audit": parsed_output.audit.model_dump() if parsed_output.audit else {},
        "request_increment_correction_attempt": True,
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
