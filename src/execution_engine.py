"""Streaming Execution Engine for Orchestral Machine.

Streams compact, structured state events plus final results/error events.
"""

import contextvars
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from src.config import RECURSION_LIMIT
from src.graph import app
from src.state import GraphState

logger = logging.getLogger(__name__)
current_task_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_task_id", default="UNKNOWN"
)


def _extract_role(node_name: Any) -> str:
    """Derive canonical role label from an audit_log node name."""
    if not node_name:
        return "SYSTEM"
    raw = str(node_name).split(".")[-1].replace("_node", "").upper()
    if raw.startswith("CORRECTOR"):
        return raw
    if raw.startswith("ARCHIVIST"):
        return "ARCHIVIST"
    return raw


def derive_state_summary(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build a lightweight summary from graph state for external listeners."""
    status = str(state_dict.get("status") or "running")
    audit_log = state_dict.get("audit_log") or []
    last_entry = audit_log[-1] if audit_log else {}

    if isinstance(last_entry, dict):
        last_node = last_entry.get("node")
        model = last_entry.get("model_id") or state_dict.get("active_model")
    else:
        last_node = getattr(last_entry, "node", None)
        model = getattr(last_entry, "model_id", None) or state_dict.get("active_model")

    role = _extract_role(last_node)
    loop = int(state_dict.get("loop_iteration", 0) or 0)
    correction_attempt = int(state_dict.get("correction_attempt", 0) or 0)
    archival_attempt = int(state_dict.get("archival_attempt", 0) or 0)
    validation_attempt = int(state_dict.get("validation_attempt", 0) or 0)
    meta_attempt = int(state_dict.get("meta_attempt", 0) or 0)

    if role == "VALIDATOR":
        attempt = validation_attempt
    elif role == "ARCHIVIST":
        attempt = max(archival_attempt, meta_attempt)
    else:
        attempt = correction_attempt

    default_messages = {
        "plan_ready": "Plan completed.",
        "generated": "Implementation generated.",
        "PASS": "Tests passed.",
        "FAIL": "Tests failed.",
        "fixed": "Fixes applied.",
        "error_L1": "Review level L1 reported issues.",
        "error_L2": "Review level L2 reported issues.",
        "error_L3": "Review level L3 requires rewrite.",
        "execution_failure": "Verification failed execution constraints.",
        "COMPLETED": "Task completed.",
        "HIR": "Human intervention required.",
    }
    content = state_dict.get("notes") or default_messages.get(status, "State updated.")
    fallback_active = bool(state_dict.get("system_flags", {}).get("fallback_active"))
    if fallback_active:
        content = f"{content} (Reservist Active)"
    state_event_log = state_dict.get("event_log", [])
    last_event = state_event_log[-1].get("event", "") if state_event_log else ""

    return {
        "phase": status,
        "status": status,
        "role": role,
        "step": len(audit_log),
        "content": content,
        "model": model,
        "attempt": attempt,
        "loop": loop,
        "correction_attempt": correction_attempt,
        "fallback_active": fallback_active,
        "last_event": last_event,
    }


def _stable_hash(value: Any) -> str:
    """Compute deterministic hash for nested JSON-compatible values."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _persist_intermediate_artifacts(
    task_id: str,
    state_dict: Dict[str, Any],
    cache: Dict[str, str],
) -> None:
    """Persist key intermediate artifacts when they change."""
    artifacts_dir = Path("Finish") / task_id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    plan = state_dict.get("plan")
    if plan is not None:
        plan_hash = _stable_hash(plan)
        if cache.get("plan") != plan_hash:
            (artifacts_dir / "plan.json").write_text(
                json.dumps(plan, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            cache["plan"] = plan_hash

    code = state_dict.get("code") or {}
    if isinstance(code, dict) and code:
        code_hash = _stable_hash(code)
        if cache.get("code") != code_hash:
            code_dir = artifacts_dir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in code.items():
                file_path = code_dir / str(filename)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(str(content), encoding="utf-8")
            cache["code"] = code_hash

    review_payload = state_dict.get("review") or state_dict.get("review_feedback")
    if review_payload is not None:
        review_hash = _stable_hash(review_payload)
        if cache.get("review") != review_hash:
            (artifacts_dir / "review.json").write_text(
                json.dumps(review_payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            cache["review"] = review_hash

    diagnosis_payload = state_dict.get("diagnosis") or state_dict.get(
        "verifier_feedback"
    )
    if diagnosis_payload is not None:
        diagnosis_hash = _stable_hash(diagnosis_payload)
        if cache.get("diagnosis") != diagnosis_hash:
            (artifacts_dir / "diagnosis.json").write_text(
                json.dumps(
                    diagnosis_payload, indent=2, ensure_ascii=False, default=str
                ),
                encoding="utf-8",
            )
            cache["diagnosis"] = diagnosis_hash


def run_task_generator(
    task_text: str, task_id: str, resume_state: Optional[Dict[str, Any]] = None
) -> Generator[Dict[str, Any], None, None]:
    """Execute the graph with streaming support.

    This generator yields STATE events as the graph executes.

    Args:
        task_text: The task description to execute.
        task_id: Unique identifier for this task execution.

    Yields:
        STATE events with enriched context:
            {"type": "STATE", "phase": "...", "status": "...", "role": "...", ...}
        RESULT events:
            {
                "type": "RESULT",
                "payload": final_state_dict
            }
        ERROR events:
            {
                "type": "ERROR",
                "error": "error message"
            }
    """
    import re

    # --- Input Validation ---
    def _validate_task_id(tid: str) -> str:
        if not tid or not isinstance(tid, str):
            raise ValueError("task_id must be a non-empty string")
        if len(tid) > 128:
            raise ValueError("task_id must be <= 128 characters")
        if not re.match(r"^[a-zA-Z0-9_-]+$", tid):
            raise ValueError(
                f"Invalid task_id format: {tid}. Must be alphanumeric, _, or -"
            )
        return tid

    def _validate_task_text(text: str) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("task_text must be a non-empty string")
        if len(text) > 50000:
            raise ValueError("task_text must be <= 50,000 characters")
        return text

    try:
        _validate_task_id(task_id)
        _validate_task_text(task_text)
    except ValueError as e:
        logger.error(f"Input validation failed: {e}")
        yield {"type": "ERROR", "error": str(e)}
        return

    from src.integrations.logging_ops import (
        configure_session_logging,
        teardown_session_logging,
    )

    session_handler = configure_session_logging(task_id)
    token = current_task_id.set(task_id)
    logger.info(f"Starting streaming execution for task_id={task_id}")

    # ------------------------------------------------------------------
    # [v0.6 preflight] Docker sandbox — fail fast if unavailable
    # ------------------------------------------------------------------
    _sandbox = None
    try:
        from src.sandbox import DockerSandbox, register_sandbox

        _sandbox = DockerSandbox(task_id)
        _sandbox.start()
        register_sandbox(task_id, _sandbox)
    except Exception as _sandbox_exc:
        import docker.errors as _docker_errors

        if isinstance(_sandbox_exc, _docker_errors.ImageNotFound):
            _preflight_msg = (
                "Docker image 'orchestral-worker:latest' not found. "
                "Build it first: docker build -t orchestral-worker:latest ."
            )
        elif isinstance(_sandbox_exc, _docker_errors.DockerException):
            _preflight_msg = (
                "Docker daemon is not running or not accessible. "
                "Start Docker Desktop and try again."
            )
        else:
            _preflight_msg = f"Docker sandbox could not start: {_sandbox_exc}"

        logger.critical("Pre-flight FAILED for task %s: %s", task_id, _preflight_msg)
        yield {"type": "ERROR", "error": f"Pre-flight check failed: {_preflight_msg}"}
        return

    # 1. Initialize GraphState (or use resume state)
    if resume_state is not None:
        state_dict = resume_state
        logger.info("Resuming task from provided state (task_id=%s)", task_id)
    else:
        initial_state = GraphState(task=task_text, task_id=task_id)
        state_dict = initial_state.model_dump()

    # 2. Prepare LangGraph config
    config = {"recursion_limit": RECURSION_LIMIT}
    current_state_values: Dict[str, Any] = state_dict
    last_phase: Optional[str] = None
    artifact_cache: Dict[str, str] = {}

    try:
        # 3. Execute with stream() using stream_mode="values"
        # This yields the FULL state after each step, avoiding need for get_state() (which requires checkpointer)
        for state_chunk in app.stream(state_dict, config, stream_mode="values"):
            # Verify we have a dict-like object
            if isinstance(state_chunk, dict):
                current_state_values = state_chunk
            elif hasattr(state_chunk, "values"):
                current_state_values = state_chunk.values
            elif hasattr(state_chunk, "model_dump"):
                current_state_values = state_chunk.model_dump()
            else:
                logger.warning(f"Unexpected state chunk type: {type(state_chunk)}")
                continue

            # 4. Skip partial/empty initialization chunks
            if not current_state_values.get("task"):
                continue

            summary = derive_state_summary(current_state_values)
            if summary["phase"] != last_phase:
                _persist_intermediate_artifacts(
                    task_id, current_state_values, artifact_cache
                )
                last_phase = summary["phase"]

            # 5. Yield simplified, enriched state event
            event: Dict[str, Any] = {
                "type": "STATE",
                "payload": current_state_values,
                "phase": summary["phase"],
                "role": summary["role"],
                "step": summary["step"],
                "content": summary["content"],
                "status": summary["status"],
                "model": summary["model"],
                "attempt": summary["attempt"],
                "loop": summary["loop"],
                "correction_attempt": summary["correction_attempt"],
                "fallback_active": summary.get("fallback_active", False),
                "last_event": summary.get("last_event", ""),
            }
            yield event

        # 6. Final result is the last processed chunk (current_state_values)
        # In stream_mode="values", the loop finishes with the final state.
        logger.info(f"Graph execution completed for task_id={task_id}")

        yield {"type": "RESULT", "payload": current_state_values}

    except Exception as e:
        logger.exception(f"Graph execution failed for task_id={task_id}")
        yield {"type": "ERROR", "error": str(e)}
    finally:
        current_task_id.reset(token)
        teardown_session_logging(session_handler)
        if _sandbox is not None:
            from src.sandbox import unregister_sandbox

            try:
                _sandbox.stop()
            except Exception as _stop_exc:
                logger.error(
                    "Failed to stop Docker sandbox for task %s: %s", task_id, _stop_exc
                )
            unregister_sandbox(task_id)
            from src.config import SANDBOX_CLEANUP_WORKSPACES

            if SANDBOX_CLEANUP_WORKSPACES:
                import shutil

                try:
                    shutil.rmtree(_sandbox.workspace_path, ignore_errors=True)
                except Exception as _clean_exc:
                    logger.warning(
                        "Failed to clean workspace for task %s: %s",
                        task_id,
                        _clean_exc,
                    )


def run_task_simple(
    task_text: str, task_id: str, resume_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Simple wrapper for blocking execution (legacy compatibility).

    This function provides a simple interface for cases where streaming
    is not required, maintaining compatibility with existing code.

    Args:
        task_text: The task description to execute.
        task_id: Unique identifier for this task execution.

    Returns:
        The final state dictionary after execution.
    """
    result = {"status": "ERROR", "error": "No events received"}

    for event in run_task_generator(task_text, task_id, resume_state=resume_state):
        if event["type"] == "RESULT":
            result = event["payload"]
        elif event["type"] == "ERROR":
            result = {"status": "ERROR", "error": event["error"]}

    return result
