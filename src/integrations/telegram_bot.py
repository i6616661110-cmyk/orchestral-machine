"""Telegram Bot Client for Orchestral Machine.

This module provides a raw API client for the Telegram Bot API using the
`requests` library. It implements send_message, edit_message, and get_updates
methods for integration with the Orchestral Machine.
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram Bot API client.

    Provides methods to interact with the Telegram Bot API using raw HTTP
    requests. This avoids external dependencies like python-telegram-bot.
    """

    def __init__(self, token: str) -> None:
        """Initialize the Telegram Bot client.

        Args:
            token: The Telegram Bot API token.
        """
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}/"

    def _request(self, method: str, payload: Optional[dict] = None) -> Optional[dict]:
        """Make a request to the Telegram API.

        Args:
            method: The API method name (e.g., 'sendMessage').
            payload: Optional dictionary of parameters to send.

        Returns:
            The 'result' field from the Telegram API response, or None on error.
        """
        url = f"{self.base_url}{method}"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data.get("ok"):
                    logger.error(f"Telegram API error: {data.get('description')}")
                    return None

                return data.get("result")

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Telegram API request failed (attempt %d/%d): %s. Retrying...",
                        attempt + 1,
                        max_retries,
                        e,
                    )
                    time.sleep(1)
                else:
                    logger.error(
                        "Telegram API request failed after %d attempts: %s",
                        max_retries,
                        e,
                    )
                    return None
            except (ValueError, KeyError) as e:
                logger.error(f"Error parsing Telegram API response: {e}")
                return None

    def send_message(
        self, chat_id: str, text: str, parse_mode: Optional[str] = None
    ) -> Optional[dict]:
        """Send a message to a chat.

        Args:
            chat_id: The target chat ID.
            text: The message text.
            parse_mode: Optional parse mode ('Markdown' or 'HTML').

        Returns:
            The sent message object (contains 'message_id'), or None on error.
        """
        max_len = 4096
        if len(text) > max_len:
            text = text[: max_len - 20] + "\n...[truncated]"

        payload: dict = {
            "chat_id": chat_id,
            "text": text,
        }

        if parse_mode:
            payload["parse_mode"] = parse_mode

        result = self._request("sendMessage", payload)

        if result is None and parse_mode:
            logger.warning("Retrying message without parse_mode (Markdown fallback)")
            payload.pop("parse_mode", None)
            result = self._request("sendMessage", payload)

        if result:
            logger.info(
                f"Message sent to chat {chat_id}, message_id: {result.get('message_id')}"
            )

        return result

    def edit_message(
        self, chat_id: str, message_id: int, text: str, parse_mode: Optional[str] = None
    ) -> Optional[dict]:
        """Edit an existing message.

        Args:
            chat_id: The chat ID containing the message.
            message_id: The message ID to edit.
            text: The new message text.
            parse_mode: Optional parse mode ('Markdown' or 'HTML').

        Returns:
            The edited message object, or None on error.
        """
        payload: dict = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }

        if parse_mode:
            payload["parse_mode"] = parse_mode

        result = self._request("editMessageText", payload)

        if result:
            logger.info(f"Message edited in chat {chat_id}, message_id: {message_id}")

        return result

    def get_updates(
        self, offset: Optional[int] = None, timeout: int = 30
    ) -> list[dict]:
        """Get updates from the Telegram Bot.

        Args:
            offset: The update ID to start from (exclusive).
            timeout: The long polling timeout in seconds.

        Returns:
            A list of update objects, or empty list on error.
        """
        payload: dict = {
            "timeout": timeout,
        }

        if offset is not None:
            payload["offset"] = offset

        result = self._request("getUpdates", payload)

        if result is None:
            logger.warning("Failed to get updates from Telegram")
            return []

        logger.debug(f"Received {len(result)} updates")
        return result
