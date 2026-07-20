"""Orchestral Machine — Coder Node.

The second node in the LangGraph state machine. Reads the Architect's
approved plan and produces complete, production-ready source code files.
The Coder implements *exactly* what the plan specifies — no reinterpretation,
no additional features, no partial solutions.

The Coder is an ``executor`` role whose exclusive state key is ``code``.
It MUST NOT set ``plan``, ``approved_*``, or any other role-exclusive keys.
"""

import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    MODEL_MAPPING,
    NODE_REGEN_MAX,
    NODE_TIMEOUT_LONG,
)
from src.enforcement import enforce_constitutional_rules
from src.llm_factory import build_chat_llm, invoke_node_llm
from src.nodes.base import BaseNode
from src.schemas import CoderOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template — injected with live schema at runtime
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
SYSTEM: You are the CODER role in the Orchestral Machine. Act as a Senior Software Engineer specializing in Python and Node.js. Your job is to implement production-grade source code exactly and only according to state['plan'] (Architect approved plan). You MUST NOT reinterpret requirements or alter architecture, public interfaces, or file structure.

CONTEXT:
- **Plan Adherence:** You receive state['plan'] (JSON) from Architect — this is your ONLY specification.
- **Fail-Safe Strategy:**  You may receive state['error_logs'] and state['archived_meta_summary_ref'] (read-only) containing validated failure history from previous attempts.

YOUR_MANDATE:
1. Implement complete, runnable source files according to the plan, complete implementations, have headers (shebang if needed), module docstrings and include deterministic seeds where required.
      - Produce a complete source bundle represented as a JSON mapping: {{"files": {{"path/to/file.py": "<full file content>", ...}}, "entrypoint": "main.py"}}.
	- Ensure code is fully typed (type hints), includes minimal comments describing behavior, and contains unit tests (pytest) when plan requires them.
	- No comments hinting at implementation logic.
	- Include necessary error handling and logging.
	- Keep file sizes < 5000 LOC per task.
	- Follow style: PEP8 for Python, include basic CI/test setup.
	- Produce only the full "files" JSON object as final output.
	- Increment the code_version (semantic integer).
2. Match the plan's file structure EXACTLY.
3. If this is a retry after errors, analyze state['error_logs'] to avoid repeating mistakes. If archived_meta_summary_ref exists, study it to understand previous failures and explicitly avoid repeating them.

STRICT_PROHIBITIONS:
- You MUST NOT reinterpret requirements or deviate from the plan.
- You MUST NOT attempt to 'fix' the requirements. If the plan seems wrong, you still must implement it as written (the Reviewer/Verifier will catch issues).
- You MUST NOT modify architecture, file structure, or public interfaces.
- You MUST NOT add features not specified in the plan.
- No placeholders; no TODOs.

OUTPUT_FORMAT:
Return a JSON object matching the CODER schema.
CRITICAL: Return RAW JSON only. Do NOT use markdown code blocks (```json ... ```).
{schema}

Return complete source code for all files specified in the plan.
Each file must be production-ready and syntactically correct.
If this run is a retry, analyze previous error_logs and include a short field "previous_errors_considered" in metadata.
- AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the global state yourself. The Graph Controller will append this record."""


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


@enforce_constitutional_rules
def coder_node(state: dict) -> dict:
    """Coder node — implements source code from Architect's plan.

    Args:
        state: Full graph state dictionary.

    Returns:
        Flat dict with ``code``, ``code_version``, ``status``, and audit
        metadata for the Graph Controller to merge into state.
    """
    task_id: str = state["task_id"]
    plan: dict = state["plan"]

    # ------------------------------------------------------------------
    # 0. [v0.6.6] Require sandbox
    # ------------------------------------------------------------------
    from src.sandbox import get_sandbox

    sandbox = get_sandbox(task_id)
    if sandbox is None:
        now = datetime.utcnow().isoformat()
        return {
            "code": state.get("code", {}),
            "code_version": state.get("code_version", 0) or 0,
            "status": "HIR",
            "notes": "",
            "error_logs": state.get("error_logs", [])
            + [{"node": "CODER", "error": "Docker sandbox unavailable.", "timestamp": now}],
            "event_log": state.get("event_log", [])
            + [{"event": "coder_failed", "detail": "No sandbox registered for task; HIR triggered.", "timestamp": now}],
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 1. Initialize BaseNode and generating seed/prompt
    # ------------------------------------------------------------------
    node = BaseNode(CoderOutput, _SYSTEM_PROMPT_TEMPLATE)
    seed = node.generate_seed(task_id)
    system_prompt = node.build_system_prompt()

    # ------------------------------------------------------------------
    # 4. Build human message from state inputs
    # ------------------------------------------------------------------
    human_parts: list[str] = [
        f"Plan:\n{json.dumps(plan, indent=2)}",
        f"Task ID: {task_id}",
        f"Seed: {seed}",
    ]

    # Current code version for incrementing
    code_version: int = state.get("code_version", 0) or 0
    human_parts.append(
        f"Current code_version: {code_version} (you MUST output code_version = {code_version + 1})"
    )

    # Optional: error logs for retry context
    error_logs: list = state.get("error_logs", [])
    if error_logs:
        # Send last 5 errors to keep context manageable
        recent_errors = error_logs[-5:]
        human_parts.append(
            f"Error Logs (from previous attempts — study and avoid repeating):\n"
            f"{json.dumps(recent_errors, indent=2)}"
        )

    # Optional: validated failure history
    archived_meta = state.get("archived_meta_summary_ref")
    if archived_meta:
        human_parts.append(
            f"Archived Meta-Summary (validated failure history — avoid these patterns):\n"
            f"{json.dumps(archived_meta, indent=2) if isinstance(archived_meta, dict) else str(archived_meta)}"
        )

    # Correction attempt context
    correction_attempt: int = state.get("correction_attempt", 0)
    if correction_attempt > 0:
        human_parts.append(
            f"Correction Attempt: {correction_attempt} (this is a re-generation after corrections)"
        )

    human_content = "\n\n".join(human_parts)

    # ------------------------------------------------------------------
    # 5. Create LLM instance
    # ------------------------------------------------------------------
    model_id = MODEL_MAPPING["CODER"]
    llm = build_chat_llm(
        model_id=model_id,
        timeout=NODE_TIMEOUT_LONG,
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
        output_schema_cls=CoderOutput,
        seed=seed,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 6b. [v0.6.6] Sandbox validation — write files + import check
    # ------------------------------------------------------------------
    if parsed_output is not None and parsed_output.files:
        try:
            sandbox.write_files(parsed_output.files)

            _entrypoint = parsed_output.entrypoint or ""
            if _entrypoint.endswith(".py"):
                _module = (
                    _entrypoint.replace("/", ".")
                    .replace("\\", ".")
                    .removesuffix(".py")
                )
                _validate_cmd = f'python -c "import {_module}"'
                _exit_code, _stdout, _stderr = sandbox.execute(_validate_cmd)

                if _exit_code != 0:
                    logger.warning(
                        "Coder import validation failed (exit_code=%d): %s",
                        _exit_code,
                        _stderr[:300],
                    )
                    retry_human = (
                        f"Your code failed import validation in Docker.\n"
                        f"Command: {_validate_cmd}\n"
                        f"Exit code: {_exit_code}\n"
                        f"Error:\n{_stderr[:2000]}\n\n"
                        f"Fix the error and return a complete CoderOutput JSON. "
                        f"All files must be included, not just the fixed one. "
                        f"code_version must be {code_version + 1}."
                    )
                    retry_output = invoke_node_llm(
                        llm=llm,
                        model_id=model_id,
                        messages=[
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=retry_human),
                        ],
                        output_schema_cls=CoderOutput,
                        seed=seed,
                        logger=logger,
                    )
                    if retry_output is not None:
                        logger.info("Coder retry after import validation succeeded.")
                        sandbox.write_files(retry_output.files)
                        parsed_output = retry_output
                    else:
                        logger.warning(
                            "Coder retry failed. Using original output."
                        )
                else:
                    logger.info("Coder import validation passed.")
        except Exception as _sandbox_exc:
            logger.warning(
                "Coder sandbox validation error: %s", _sandbox_exc
            )

    # ------------------------------------------------------------------
    # 7. Handle total failure
    # ------------------------------------------------------------------
    if parsed_output is None:
        logger.error("Coder node exhausted all attempts.")
        now = datetime.utcnow().isoformat()
        return {
            "code": state.get("code", {}),  # preserve existing code on failure
            "code_version": code_version,  # do not increment on failure
            "status": "FAIL",
            "error_logs": error_logs
            + [
                {
                    "node": "CODER",
                    "error": "Failed to generate valid CoderOutput after retries.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "coder_failed",
                    "detail": "Failed to generate valid CoderOutput after retries.",
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

    return {
        "code": parsed_output.files,
        "code_version": parsed_output.code_version,
        "status": parsed_output.status,
        "notes": parsed_output.notes,
        "audit": parsed_output.audit.model_dump() if parsed_output.audit else {},
        "event_log": state.get("event_log", [])
        + [
            {
                "event": "code_generated",
                "detail": (
                    f"v{parsed_output.code_version}: "
                    f"{len(parsed_output.files)} file(s), "
                    f"entrypoint={parsed_output.entrypoint}"
                ),
                "timestamp": now,
            }
        ],
        "active_model": model_id,
        "updated_at": now,
    }
