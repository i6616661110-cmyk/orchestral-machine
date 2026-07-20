"""Orchestral Machine — LLM Factory.

Centralized LLM client creation for all nodes.
Eliminates duplicate code and ensures consistent API key handling.
"""

from __future__ import annotations

import datetime
import os
import json
import logging
import re
import threading
from typing import Any, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from src.config import BASE_URL, GEN_CONFIG, NODE_REGEN_MAX
from src.integrations.logging_ops import APIRecorder


_llm_thread_stats = threading.local()

_rejected_params: dict[str, set[str]] = {}
_rejected_params_lock = threading.Lock()


def get_llm_stats() -> dict:
    """Return LLM call statistics from the current thread.

    Set by invoke_node_llm after each invocation.
    Read by _wrap_node in graph.py to enrich state updates.
    """
    return getattr(_llm_thread_stats, "stats", {})


def get_api_key() -> str:
    """Get OpenRouter API key from environment.

    Returns:
        The API key string.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment. "
            "Get your free API key at https://openrouter.ai/settings/keys"
        )
    return api_key


def build_chat_llm(
    model_id: str,
    timeout: int | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> ChatOpenAI:
    """Build a ChatOpenAI client configured for OpenRouter.

    Args:
        model_id: OpenRouter model identifier (e.g., "openai/gpt-4o").
        timeout: Request timeout in seconds. Defaults to NODE_TIMEOUT_LONG.
        seed: Integer seed for deterministic output. Must fit in 32-bit signed int.
        temperature: Generation temperature. Defaults to GEN_CONFIG["temperature"].
        top_p: Nucleus sampling parameter. Defaults to GEN_CONFIG["top_p"].
        frequency_penalty: Frequency penalty. Defaults to GEN_CONFIG["frequency_penalty"].
        presence_penalty: Presence penalty. Defaults to GEN_CONFIG["presence_penalty"].

    Returns:
        Configured ChatOpenAI instance ready for OpenRouter API.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    api_key = get_api_key()

    # Use defaults from GEN_CONFIG if not specified
    final_temperature = (
        temperature if temperature is not None else GEN_CONFIG["temperature"]
    )
    final_top_p = top_p if top_p is not None else GEN_CONFIG["top_p"]
    final_frequency_penalty = (
        frequency_penalty
        if frequency_penalty is not None
        else GEN_CONFIG["frequency_penalty"]
    )
    final_presence_penalty = (
        presence_penalty
        if presence_penalty is not None
        else GEN_CONFIG["presence_penalty"]
    )

    # Build kwargs
    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "api_key": api_key,
        "base_url": BASE_URL,
        "temperature": final_temperature,
        "top_p": final_top_p,
    }
    if final_frequency_penalty:
        llm_kwargs["frequency_penalty"] = final_frequency_penalty
    if final_presence_penalty:
        llm_kwargs["presence_penalty"] = final_presence_penalty

    # Add timeout if specified
    if timeout is not None:
        llm_kwargs["timeout"] = timeout

    # Add seed if specified (must be 32-bit signed integer)
    if seed is not None:
        llm_kwargs["seed"] = seed % (2**31)

    with _rejected_params_lock:
        rejected = frozenset(_rejected_params.get(model_id, set()))
    for param in rejected:
        llm_kwargs.pop(param, None)

    return ChatOpenAI(**llm_kwargs)


# Convenience functions for common use cases
def build_default_llm(
    model_key: str, timeout: int | None = None, seed: int | None = None
) -> ChatOpenAI:
    """Build LLM using MODEL_MAPPING from config.

    Args:
        model_key: Key in MODEL_MAPPING (e.g., "ARCHITECT", "CODER").
        timeout: Request timeout in seconds.
        seed: Integer seed for deterministic output.

    Returns:
        Configured ChatOpenAI instance.
    """
    # Import here to avoid circular imports
    from src.config import MODEL_MAPPING

    model_id = MODEL_MAPPING.get(model_key)
    if not model_id:
        raise ValueError(
            f"Unknown model key: {model_key}. Available keys: {list(MODEL_MAPPING.keys())}"
        )

    return build_chat_llm(model_id, timeout=timeout, seed=seed)


def _extract_json(text: str) -> str:
    """Extract the first complete JSON object from LLM output.

    Uses balanced-brace matching with string awareness to correctly
    handle trailing API metadata or concatenated objects.
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return text


def invoke_node_llm(
    llm: BaseChatModel,
    model_id: str,
    messages: list[BaseMessage],
    output_schema_cls: Type[BaseModel],
    seed: str,
    logger: logging.Logger,
    fallback_model_id: str | None = None,
) -> BaseModel | None:
    """Invoke LLM with retry loop, JSON validation, and error handling.

    Args:
        llm: Configured ChatOpenAI instance.
        model_id: The model identifier (for logging).
        messages: List of messages to send.
        output_schema_cls: The Pydantic model class to validate against.
        seed: The deterministic seed used (for logging).
        logger: Logger instance for the calling node.
        fallback_model_id: Optional backup model ID to use for the final attempt.

    Returns:
        Validated Pydantic model instance, or None if all retries fail.
    """
    attempts = 0
    max_attempts = NODE_REGEN_MAX + 1
    _stats = {
        "actual_model": model_id,
        "primary_model": model_id,
        "validation_errors": 0,
        "total_attempts": 0,
        "reservist_used": False,
        "last_error": "",
    }

    from src.execution_engine import current_task_id

    tid = current_task_id.get()

    # Work on a copy of messages to avoid mutating the original list during retries
    current_messages = messages.copy()

    # Track the current LLM instance (may change if fallback is used)
    current_llm = llm
    current_model_id = model_id
    _param_grace_used = False

    while attempts < max_attempts:
        attempts += 1
        _stats["total_attempts"] = attempts

        # Check if this is the final attempt
        if attempts == max_attempts:
            # If no fallback is provided, use the global RESERVIST
            if not fallback_model_id:
                # Import here to avoid circular imports at top level
                from src.config import MODEL_MAPPING

                fallback_model_id = MODEL_MAPPING["RESERVIST"]
                logger.warning(
                    f"Auto-switching to global RESERVIST model {fallback_model_id} for final attempt."
                )

            if fallback_model_id:
                logger.warning(
                    f"Primary model {model_id} failed {attempts - 1} times. "
                    f"Switching to RESERVIST {fallback_model_id} for final attempt."
                )
            # Re-build the LLM with the fallback model, keeping same params
            # We assume the original llm has these attributes available or we reconstruct
            # Since we can't easily extract params from the instance, we rely on defaults
            # or we'd need to pass params in.
            # However, looking at the call sites, `llm` is passed in.
            # To do this correctly without passing all params again, we might need to
            # assume standard build_chat_llm usage.
            # Let's try to infer or re-use the build function if we can't extract.
            # Actually, `build_chat_llm` uses defaults from GEN_CONFIG.
            # We'll just build a new one with defaults + the fallback ID.
            # We don't have the original timeout/seed here easily unless we parse them from the
            # original LLM object or pass them in.
            # The spec says "Status: Logic to Re-instantiate... using the *same* parameters".
            # The `llm` object (ChatOpenAI) stores these.

            # Extract params from current_llm to preserve them
            original_timeout = getattr(current_llm, "request_timeout", None)
            # ChatOpenAI puts model_kwargs in `model_kwargs` usually, but temperature etc are top level.
            original_temp = getattr(current_llm, "temperature", None)
            original_top_p = getattr(current_llm, "model_kwargs", {}).get("top_p")

            # Re-build LLM for fallback
            # We need to import build_chat_llm inside to avoid circular deps if it wasn't already available,
            # but it is defined in this file, so we can call it.
            current_llm = build_chat_llm(
                model_id=fallback_model_id,
                timeout=original_timeout,  # type: ignore
                temperature=original_temp,  # type: ignore
                top_p=original_top_p,  # type: ignore
                # seed is harder to extract if it was passed in model_kwargs.
                # But we have `seed` argument to this function!
                seed=int(seed, 16) % (2**31)
                if seed and all(c in "0123456789abcdef" for c in seed)
                else None,
            )
            current_model_id = fallback_model_id
            _stats["reservist_used"] = True
            _stats["actual_model"] = fallback_model_id

        logger.info(
            f"Attempt {attempts}/{max_attempts} invoking {current_model_id} (seed={seed})"
        )

        try:
            response = current_llm.invoke(current_messages)
            content = response.content
            content_text = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False, default=str)
            )

            def _serialize_message(message: BaseMessage) -> dict[str, Any]:
                if hasattr(message, "model_dump"):
                    return message.model_dump()  # type: ignore[no-any-return]
                if hasattr(message, "dict"):
                    return message.dict()  # type: ignore[no-any-return]
                return {
                    "type": type(message).__name__,
                    "content": getattr(message, "content", str(message)),
                }

            if hasattr(response, "model_dump"):
                response_payload: Any = response.model_dump()
            elif hasattr(response, "dict"):
                response_payload = response.dict()
            else:
                response_payload = {"type": type(response).__name__, "content": content}

            record: dict[str, Any] = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "task_id": tid,
                "node": logger.name,
                "model": current_model_id,
                "inputs": [_serialize_message(m) for m in current_messages],
                "outputs": response_payload,
                "usage": getattr(response, "usage_metadata", None),
                "attempt": attempts,
            }
            APIRecorder.append(tid, record)

            # 1. Try to validate JSON against schema
            try:
                cleaned = _extract_json(content_text)
                if cleaned != content_text:
                    logger.debug(
                        "JSON extracted from wrapped output (%d -> %d chars)",
                        len(content_text),
                        len(cleaned),
                    )
                validated_output = output_schema_cls.model_validate_json(cleaned)
                logger.info("Output validated successfully.")
                _stats["actual_model"] = current_model_id
                _stats["total_attempts"] = attempts
                _llm_thread_stats.stats = _stats
                return validated_output

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Validation failed on attempt {attempts}: {e}")
                _stats["validation_errors"] += 1
                _stats["last_error"] = str(e)[:100]

                if attempts < max_attempts:
                    # FEEDBACK LOOP: Add error message and ask for correction
                    current_messages.append(response)
                    error_msg = (
                        f"Your output failed validation. Error: {e!s}. "
                        "Please regenerate the JSON strictly following the schema."
                    )
                    current_messages.append(HumanMessage(content=error_msg))
                else:
                    logger.error("Max retries reached. Validation failed.")
                    _stats["fatal"] = True
                    _llm_thread_stats.stats = _stats
                    return None

        except Exception as e:
            logger.error(f"LLM invocation error on attempt {attempts}: {e}")
            _stats["last_error"] = str(e)[:100]

            if "unexpected keyword argument" in str(e):
                match = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                rejected_param = match.group(1) if match else "seed"
                with _rejected_params_lock:
                    _rejected_params.setdefault(current_model_id, set()).add(
                        rejected_param
                    )
                logger.warning(
                    "Model %s rejected parameter '%s'. Cached for this session. Rebuilding LLM.",
                    current_model_id,
                    rejected_param,
                )
                _stats["last_error"] = (
                    f"Param '{rejected_param}' rejected by "
                    f"{current_model_id.split('/')[-1]}"
                )
                current_llm = build_chat_llm(
                    model_id=current_model_id,
                    timeout=getattr(current_llm, "request_timeout", None),
                    temperature=getattr(current_llm, "temperature", None),
                    seed=int(seed, 16) % (2**31)
                    if seed and all(c in "0123456789abcdef" for c in seed)
                    else None,
                )
                if attempts >= max_attempts and not _param_grace_used:
                    max_attempts += 1
                    _param_grace_used = True
                    logger.info(
                        "Param rejection on last attempt for %s: granting one extra attempt.",
                        current_model_id,
                    )

            if attempts >= max_attempts:
                _stats["fatal"] = True
                _llm_thread_stats.stats = _stats
                return None

    _stats["fatal"] = True
    _llm_thread_stats.stats = _stats
    return None
