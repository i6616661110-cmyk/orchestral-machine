"""Telegram Log Handler for Orchestral Machine.

This module provides a custom logging handler that pushes log records to a
queue. This allows the main bot loop to consume logs and batch-send them to
Telegram, avoiding network calls inside the logging thread.
"""

from __future__ import annotations

import datetime
import json
import logging
import queue
import re
import threading
from pathlib import Path
from typing import Any


class SensitiveDataFilter(logging.Filter):
    """Filter that sanitizes sensitive information from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive patterns from the log message."""
        if not isinstance(record.msg, str):
            return True

        # Redaction patterns
        patterns = [
            # OpenAI/OpenRouter Keys (sk-...)
            (r"sk-[a-zA-Z0-9-]{20,}", "[REDACTED]"),
            # Generic secrets (token=..., key: ...)
            (r"(?i)\b(token|key|secret|password|passwd|api[_-]?key)\s*[:=]\s*[^\s]+", r"\1=[REDACTED]"),
            # Seeds
            (r"(?i)Seed:\s*\d+", "Seed: [REDACTED]"),
        ]

        for pattern, replacement in patterns:
            record.msg = re.sub(pattern, replacement, record.msg)
            
        return True


class LogNoiseFilter(logging.Filter):
    """Filter that suppresses low-level debug/info logs from noisy libraries.
    
    This filter prevents flooding the Telegram channel with internal HTTP logs
    and other verbose output that is not relevant for high-level monitoring.
    It suppresses logs from specific loggers unless they are at ERROR level.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Suppress logs from noisy libraries below ERROR level."""
        # List of logger prefixes to suppress
        noisy_loggers = (
            "httpx",
            "httpcore",
            "openai",
            "langchain",
        )
        
        # Check if the record comes from a noisy logger
        if record.name.startswith(noisy_loggers):
            # Allow only if it is an ERROR or CRITICAL
            if record.levelno >= logging.ERROR:
                return True
            # Suppress (return False)
            return False
            
        # Allow all other logs
        return True


class TelegramLogHandler(logging.Handler):
    """Custom logging handler that pushes formatted logs to a queue.
    
    This handler formats log records and puts them into a queue for later
    processing by the bot's main loop. This is essential for:
    - Avoiding blocking network calls from the logging thread
    - Batching log messages to reduce API calls
    - Thread-safe log handling
    """

    def __init__(self, log_queue: queue.Queue) -> None:
        """Initialize the handler with a queue.
        
        Args:
            log_queue: A queue.Queue instance to push formatted logs into.
        """
        super().__init__()
        self.log_queue = log_queue
        # Attach sensitive data filter by default
        self.addFilter(SensitiveDataFilter())
        # Attach noise filter to keep Telegram channel clean
        self.addFilter(LogNoiseFilter())

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a formatted log record to the queue.
        
        Args:
            record: The log record to format and queue.
        """
        try:
            # Anti-recursion: ignore all logs from the bot's own operations
            if record.name.startswith("src.integrations"):
                return

            # Format the record using the handler's formatter
            # Default format: Time + Level + Message (like terminal output)
            formatted = self.format(record)
            
            # Push to queue (non-blocking)
            try:
                self.log_queue.put_nowait(formatted)
            except queue.Full:
                # Queue is full, drop this log message
                pass
                
        except Exception:
            # Handle any errors in formatting/queuing
            self.handleError(record)

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record.
        
        Args:
            record: The log record to format.
            
        Returns:
            Formatted string representation of the record.
        """
        # If a formatter is set, use it
        if self.formatter:
            return self.formatter.format(record)
            
        # Default format: Time + Level + Message (terminal-like)
        # We must use a Formatter to ensure record.asctime is generated
        default_formatter = logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s", datefmt="%H:%M:%S")
        return default_formatter.format(record)


_SESSION_LOCK = threading.Lock()
_ACTIVE_SESSION_HANDLER: tuple[str, logging.FileHandler] | None = None


def _normalize_task_id(task_id: str | None) -> str:
    """Return a safe task_id for file-system paths."""
    if not task_id:
        return "UNKNOWN"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(task_id))
    return safe or "UNKNOWN"


def _session_dir(task_id: str | None) -> Path:
    """Get Finish/<task_id> directory and ensure it exists."""
    safe_task_id = _normalize_task_id(task_id)
    path = Path("Finish") / safe_task_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_session_logging(task_id: str) -> logging.FileHandler:
    """Attach a per-task file handler to root logger and return it."""
    global _ACTIVE_SESSION_HANDLER
    root_logger = logging.getLogger()
    session_file = _session_dir(task_id) / "terminal.log"

    with _SESSION_LOCK:
        if _ACTIVE_SESSION_HANDLER:
            _, existing = _ACTIVE_SESSION_HANDLER
            if existing in root_logger.handlers:
                root_logger.removeHandler(existing)
            existing.close()
            _ACTIVE_SESSION_HANDLER = None

        handler = logging.FileHandler(session_file, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root_logger.addHandler(handler)
        _ACTIVE_SESSION_HANDLER = (_normalize_task_id(task_id), handler)
        return handler


def teardown_session_logging(handler: logging.Handler | None) -> None:
    """Detach and close a previously registered session file handler."""
    global _ACTIVE_SESSION_HANDLER
    if handler is None:
        return

    root_logger = logging.getLogger()
    with _SESSION_LOCK:
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

        if _ACTIVE_SESSION_HANDLER and _ACTIVE_SESSION_HANDLER[1] is handler:
            _ACTIVE_SESSION_HANDLER = None


class APIRecorder:
    """Thread-safe JSONL recorder for model API interactions."""

    _locks: dict[str, threading.Lock] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def _lock_for_task(cls, task_id: str) -> threading.Lock:
        with cls._registry_lock:
            if task_id not in cls._locks:
                cls._locks[task_id] = threading.Lock()
            return cls._locks[task_id]

    @classmethod
    def append(cls, task_id: str | None, record: dict[str, Any]) -> None:
        """Append one API interaction record to Finish/<task_id>/api_history.jsonl."""
        safe_task_id = _normalize_task_id(task_id)
        lock = cls._lock_for_task(safe_task_id)
        path = _session_dir(safe_task_id) / "api_history.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str)

        try:
            with lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            logging.getLogger(__name__).exception("Failed to append API history for task_id=%s", safe_task_id)


def log_chat_event(
    task_id: str | None,
    sender: str,
    message: str,
    *,
    chat_id: str | None = None,
    user_id: str | int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one Telegram chat event to Finish/<task_id>/chat_history.jsonl."""
    if task_id is None:
        lock = APIRecorder._lock_for_task("SYSTEM")
        path = Path("Finish") / "system_chat_history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload_task_id = "SYSTEM"
    else:
        safe_task_id = _normalize_task_id(task_id)
        lock = APIRecorder._lock_for_task(safe_task_id)
        path = _session_dir(safe_task_id) / "chat_history.jsonl"
        payload_task_id = safe_task_id

    payload: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task_id": payload_task_id,
        "sender": sender,
        "message": message,
    }
    if chat_id is not None:
        payload["chat_id"] = str(chat_id)
    if user_id is not None:
        payload["user_id"] = str(user_id)
    if metadata:
        payload["metadata"] = metadata

    try:
        with lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logging.getLogger(__name__).exception("Failed to append chat history for task_id=%s", payload_task_id)
