"""Telegram Listener for Orchestral Machine.

This module implements the main event loop for the Telegram Bot, connecting
all phases:
- Phase 1: TelegramBot client
- Phase 2: Streaming execution engine
- Phase 3: This listener loop

Usage:
    python main.py serve-bot
"""

import logging
import os
import queue
import threading
import time
import uuid
from typing import Dict, Optional

from dotenv import load_dotenv

from src.integrations.logging_ops import log_chat_event

logger = logging.getLogger(__name__)


class TelegramListener:
    """Main event loop for Telegram Bot integration.

    Responsibilities:
    - Poll for updates from Telegram
    - Handle commands (/start, /status)
    - Execute tasks in background threads
    - Stream logs and state updates to the chat
    """

    def __init__(self) -> None:
        """Initialize the Telegram listener."""
        # Load environment
        load_dotenv()

        # Load token
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env")

        # Initialize bot client
        from src.integrations.telegram_bot import TelegramBot

        self.bot = TelegramBot(self.token)

        # Queue for execution events
        self.event_queue: queue.Queue = queue.Queue()

        # State tracking
        self.current_task: Optional[str] = None
        self.current_task_id: Optional[str] = None
        self.current_task_thread: Optional[threading.Thread] = None
        self.last_update_id: Optional[int] = None

        # Message tracking for edits
        self.status_message_id: Optional[int] = None
        self.status_chat_id: Optional[int] = None

        self.last_status_time: float = 0.0
        self._last_status_content: Optional[tuple] = None  # Deduplication cache
        self._fallback_notified = False
        self._hir_notified = False

    def run(self) -> None:
        """Main event loop."""
        logger.info("Starting Telegram listener...")

        # Send startup message to configured chat if available
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if chat_id:
            from src.config import MODEL_MAPPING

            mapping_lines = "\n".join(
                f"• {role}: {model}" for role, model in MODEL_MAPPING.items()
            )
            self._send_message(
                chat_id,
                "🤖 *Orchestral Machine Bot is now online!*\n\n"
                "*Active Configuration:*\n"
                f"{mapping_lines}\n\n"
                "Send me a task description to begin.",
                parse_mode="Markdown",
            )

        logger.info("Telegram listener started. Waiting for messages...")

        while True:
            try:
                # Poll for updates
                self._poll_updates()

                # Process event queue
                self._process_events()

                # Check if task thread is done
                self._check_task_status()

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.exception(f"Error in main loop: {e}")

            time.sleep(1)

    def _poll_updates(self) -> None:
        """Poll Telegram for new updates."""
        updates = self.bot.get_updates(offset=self.last_update_id, timeout=5)

        for update in updates:
            update_id = update.get("update_id")

            # Update offset
            if update_id:
                self.last_update_id = update_id + 1

            # Process message
            if "message" in update:
                self._handle_message(update["message"])

    def _handle_message(self, message: Dict) -> None:
        """Handle incoming Telegram message."""
        chat_id = str(message["chat"]["id"])
        user_id = message.get("from", {}).get("id")
        raw_text = message.get("text", "")

        # --- Sanitization ---
        def sanitize_user_input(text: str) -> str:
            if not text:
                return text

            # Simple patterns to block basic prompt injection
            patterns = [
                (r"(?i)ignore previous instructions", "[FILTERED]"),
                (
                    r"(?i)^system:",
                    "[FILTERED]",
                ),  # Prevent potential role spoofing if naive parsing used
            ]

            import re

            sanitized = text
            for pattern, replacement in patterns:
                sanitized = re.sub(pattern, replacement, sanitized)
            return sanitized

        text = sanitize_user_input(raw_text)
        if text != raw_text:
            logger.warning(f"Sanitized input from {user_id}: '{raw_text}' -> '{text}'")

        # Security check
        from src.config import ALLOWED_TELEGRAM_USERS

        if user_id not in ALLOWED_TELEGRAM_USERS:
            logger.warning(
                f"Unauthorized access attempt from user_id={user_id} (chat_id={chat_id})"
            )
            return

        normalized_text = text.lower().strip()
        command_aliases = {"start", "help", "status", "stop", "reset", "halt"}
        if normalized_text in command_aliases:
            self._handle_command(chat_id, f"/{normalized_text}")
            return

        if len(text.strip()) < 15 and not text.startswith("/"):
            self._send_message(
                chat_id,
                "⚠️ Task description too short. Please describe your task in detail.",
            )
            return

        logger.info(f"Received message from {chat_id}: {text[:50]}...")

        # Handle commands
        if text.startswith("/"):
            self._handle_command(chat_id, text)
        else:
            # Treat as task
            self._handle_task(chat_id, text)

    def _handle_command(self, chat_id: str, command: str) -> None:
        """Handle bot commands."""
        log_chat_event(
            None,
            "user",
            command,
            chat_id=chat_id,
            metadata={"type": "command"},
        )
        if command == "/start":
            from src.config import MODEL_MAPPING

            mapping_lines = "\n".join(
                f"• {role}: {model}" for role, model in MODEL_MAPPING.items()
            )
            self._send_message(
                chat_id,
                "🤖 *Orchestral Machine*\n\n"
                "I am an autonomous coding factory powered by AI.\n\n"
                "*Active Configuration:*\n"
                f"{mapping_lines}\n\n"
                "Simply send me a task description and I'll execute it.\n\n"
                "Commands:\n"
                "/start - Show this message\n"
                "/status - Check current task status",
                parse_mode="Markdown",
            )

        elif command == "/status":
            if self.current_task:
                status_text = (
                    f"📋 *Current Task*\n\n"
                    f"Task: {self.current_task[:100]}...\n"
                    f"Status: Running"
                )
                self._send_message(chat_id, status_text, parse_mode="Markdown")
            else:
                from src.config import MODEL_MAPPING

                config_lines = [
                    f"• {role}: {model}" for role, model in MODEL_MAPPING.items()
                ]
                config_text = "\n".join(config_lines)
                dashboard_text = (
                    "🎹 Orchestral Machine Ready\n\n"
                    "Active Configuration:\n"
                    f"{config_text}\n\n"
                    "Send a task to begin."
                )
                self._send_message(chat_id, dashboard_text)

        else:
            self._send_message(
                chat_id, "Unknown command. Send a task description to begin."
            )

    def _handle_task(self, chat_id: str, task_text: str) -> None:
        """Handle task execution request."""
        # Check if already running
        if self.current_task:
            self._send_message(
                chat_id, "⏳ A task is already running. Please wait for it to complete."
            )
            return

        # Store current task info
        self.current_task = task_text
        self.current_task_id = uuid.uuid4().hex
        self.status_chat_id = chat_id

        # For forensics accuracy, task prompt is the first entry for this task_id.
        log_chat_event(
            self.current_task_id,
            "user",
            task_text,
            chat_id=chat_id,
            metadata={"type": "task_start"},
        )

        # Notify task start with explicit task identifier and config.
        from src.config import MODEL_MAPPING

        config_lines = "\n".join(
            f"• {role}: {model}" for role, model in MODEL_MAPPING.items()
        )

        result = self._send_message(
            chat_id,
            f"🏁 *Task Started* (ID: `{self.current_task_id[:8]}`)\n"
            f"Config:\n{config_lines}\n\n"
            f"{_truncate(task_text, 50)}",
            parse_mode="Markdown",
        )

        if result:
            self.status_message_id = result.get("message_id")

        # Start task in background thread
        self._start_task_thread(chat_id, task_text, self.current_task_id)

    def _start_task_thread(self, chat_id: str, task_text: str, task_id: str) -> None:
        """Start task execution in a background thread."""

        def worker():
            from src.execution_engine import run_task_generator

            for event in run_task_generator(task_text, task_id):
                self.event_queue.put(event)

        self.current_task_thread = threading.Thread(target=worker, daemon=True)
        self.current_task_thread.start()
        logger.info(f"Started task {task_id} in background thread")

    def _process_events(self) -> None:
        """Process events from the event queue."""
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            event_type = event.get("type")

            if event_type == "STATE":
                self._handle_state_event(event)
            elif event_type == "RESULT":
                self._handle_result_event(event)
            elif event_type == "ERROR":
                self._handle_error_event(event)

    def _handle_state_event(self, event: Dict) -> None:
        """Handle STATE event with structured status signal updates."""
        if not self.status_chat_id:
            return

        status = str(event.get("status") or event.get("phase") or "unknown")
        role = str(event.get("role") or "SYSTEM").upper()
        attempt = int(event.get("attempt") or 0)
        correction_attempt = int(event.get("correction_attempt") or 0)

        now = time.time()
        terminal_status = {"COMPLETED", "HIR", "ERROR"}
        if (now - self.last_status_time) < 1.0 and status not in terminal_status:
            return
        self.last_status_time = now

        content_hash = (
            role,
            status,
            attempt,
            correction_attempt,
            int(event.get("loop") or 0),
        )
        if content_hash == self._last_status_content:
            return
        self._last_status_content = content_hash

        fallback_active = event.get("fallback_active", False)
        if fallback_active and not self._fallback_notified:
            self._fallback_notified = True
            role = str(event.get("role") or "SYSTEM").upper()
            self._send_message(
                self.status_chat_id,
                f"⚠️ *MODEL FALLBACK*\n[{role}] Primary model failed. Switched to RESERVIST.",
                parse_mode="Markdown",
            )

        if status == "COMPLETED":
            status_text = "✅ *COMPLETED*"
        elif status == "HIR":
            if self._hir_notified:
                return
            self._hir_notified = True
            status_text = "🛑 *HUMAN INTERVENTION REQUIRED*"
        elif role == "REVIEWER" and status in {"error_L1", "error_L2"}:
            status_text = f"⚠️ *[REVIEWER]* -> `{status}`"
        elif role.startswith("CORRECTOR") and status in {"fixed", "no_change"}:
            status_text = f"🔄 *[{role}]* -> `{status}`"
        elif role == "VERIFIER" and status == "execution_failure":
            status_text = "❌ *[VERIFIER]* -> `execution_failure`"
        elif role == "REVIEWER" and status == "error_L3":
            status_text = "⚠️ *[REVIEWER]* -> `error_L3`"
        else:
            status_text = f"⚙️ *[{role}]* -> `{status}`"

        # Always show model name for operator visibility
        model = str(event.get("model") or "")
        if model:
            short_model = model.split("/")[-1] if "/" in model else model
            status_text += f"\n`{short_model}`"

        content = str(event.get("content") or "")
        last_event = event.get("last_event", "")
        notable_events = {
            "hard_reset_executed",
            "verifier_correction_reset",
            "a2_feedback_to_a1",
            "validator_feedback_to_a1",
            "validator_feedback_to_a2",
            "model_fallback_exhausted",
            "system_halted",
            "node_hard_timeout",
        }
        if last_event in notable_events:
            event_display = last_event.replace("_", " ").title()
            status_text += f"\n📋 `{event_display}`"

        # Show node notes when present
        if content and content not in status_text and "State updated." not in content:
            display_content = content[:200] + "..." if len(content) > 200 else content
            status_text += f"\n_{display_content}_"

        self._send_message(self.status_chat_id, status_text, parse_mode="Markdown")

    def _handle_result_event(self, event: Dict) -> None:
        """Handle RESULT event - send final message."""
        if not self.status_chat_id:
            return

        payload = event.get("payload", {})
        status = payload.get("status", "unknown")

        if status == "COMPLETED":
            message = "✅ *COMPLETED*"
        elif status == "HIR":
            if not self._hir_notified:
                message = "🛑 *HUMAN INTERVENTION REQUIRED*"
            else:
                message = "🛑 *Task stopped (HIR)*"
        else:
            message = f"ℹ️ *Task Finished* with status: {status}"

        # Save results to Finish/ folder
        if self.current_task_id:
            from src.integrations.persistence import save_task_results

            save_task_results(self.current_task_id, payload)

        # Send final message
        self._send_message(self.status_chat_id, message, parse_mode="Markdown")

        # Clear current task
        self.current_task = None
        self._fallback_notified = False
        self._hir_notified = False
        # self.current_task_id = None
        self.status_message_id = None

    def _handle_error_event(self, event: Dict) -> None:
        """Handle ERROR event - send error message."""
        if not self.status_chat_id:
            return

        error = event.get("error", "Unknown error")

        self._send_message(
            self.status_chat_id, f"❌ *Error*\n\n{error}", parse_mode="Markdown"
        )

        # Clear current task
        self.current_task = None
        self._fallback_notified = False
        self._hir_notified = False
        # self.current_task_id = None
        self.status_message_id = None

    def _check_task_status(self) -> None:
        """Check if task thread is done."""
        if self.current_task_thread and not self.current_task_thread.is_alive():
            logger.info(f"Task {self.current_task_id} thread finished")
            self.current_task = None
            self.current_task_id = None
            self.current_task_thread = None

    def _send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> Optional[Dict]:
        """Send bot message and persist it to chat history."""
        result = self.bot.send_message(chat_id, text, parse_mode=parse_mode)
        log_chat_event(
            self.current_task_id,
            "bot",
            text,
            chat_id=chat_id,
            metadata={"parse_mode": parse_mode} if parse_mode else None,
        )
        return result


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def run_listener() -> None:
    """Entry point for the listener."""
    listener = TelegramListener()
    listener.run()
