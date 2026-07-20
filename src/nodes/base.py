"""Orchestral Machine — Base Node Infrastructure.

This module defines the `BaseNode` class, which serves as the foundation for all
role-specific nodes in the graph. It encapsulates common logic for:
1.  Output schema management (JSON dumping).
2.  System prompt formatting.
3.  Deterministic seed generation.
"""

import hashlib
import json
from typing import Type, Any

from pydantic import BaseModel


class BaseNode:
    """Base class for all graph nodes.

    Encapsulates common functionality required by every node, ensuring consistency
    in how schemas are injected into prompts and how deterministic seeds are
    generated.
    """

    def __init__(self, output_schema: Type[BaseModel], system_prompt_template: str):
        """Initialize the node with its output schema and prompt template.

        Args:
            output_schema: The Pydantic model class defining the expected JSON output.
            system_prompt_template: A string template for the system prompt, expected
                to contain a `{schema}` placeholder.
        """
        self.output_schema = output_schema
        self.system_prompt_template = system_prompt_template
        self._cached_schema_json = json.dumps(
            self.output_schema.model_json_schema(), indent=2
        )

    def generate_seed(self, task_id: str) -> str:
        """Generate a deterministic seed string based on the task ID.

        Uses SHA-256 to derive a stable seed from the task ID, ensuring that
        re-runs with the same task ID produce consistent results (where possible).

        Args:
            task_id: The unique identifier for the current task.

        Returns:
            A hex string representing the deterministic seed.
        """
        return hashlib.sha256(task_id.encode("utf-8")).hexdigest()

    def build_schema_json(self) -> str:
        """Return cached JSON schema string.

        Returns:
            A formatted JSON string representation of the Pydantic model's schema.
        """
        return self._cached_schema_json

    def build_system_prompt(self, **kwargs: Any) -> str:
        """Format the system prompt template with the schema and other variables.

        Automatically injects the JSON schema into the `{schema}` placeholder.
        Any additional keyword arguments are passed to the format method.

        Args:
            **kwargs: Additional variables to inject into the template (e.g., `role`, `context`).

        Returns:
            The fully formatted system prompt string.
        """
        schema_json = self.build_schema_json()
        return self.system_prompt_template.format(schema=schema_json, **kwargs)
