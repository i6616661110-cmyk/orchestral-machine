"""Orchestral Machine — Constitutional Enforcement System.

Ensures that no node (role) violates its boundaries by writing to protected
state keys or keys exclusive to other roles.  Any violation immediately
triggers HIR (Human Intervention Required) with no recovery path.

Constitutional principles enforced:
    1. Only ``decision_authority`` roles (verifier, validator, architect) may
       set ``approved_*`` keys.
    2. Each role has exclusive keys that only it may write.
    3. The ``task`` field is 100 %% read-only — no node may mutate it.
    4. All violations are logged at CRITICAL level, audited, and the system
       is halted with ``status='HIR'``.

Exports:
    ConstitutionalViolation         — exception for boundary violations
    validate_state_mutation          — core validation logic
    extract_role_from_node_name      — helper to normalise node names
    emit_audit_for_violation         — audit entry factory
    enforce_constitutional_rules     — decorator applied to every node function
"""

import logging
from datetime import datetime
from functools import wraps
from typing import Any, Dict

from src.config import (
    PROTECTED_APPROVAL_KEYS,
    ROLE_CLASSIFICATION,
    ROLE_EXCLUSIVE_KEYS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ConstitutionalViolation(Exception):
    """Raised when a role violates constitutional boundaries."""


# ---------------------------------------------------------------------------
# Role extraction
# ---------------------------------------------------------------------------

def extract_role_from_node_name(node_name: str) -> str:
    """Extract the base role from a node name.

    Examples::

        "CORRECTOR_C1"  → "corrector"
        "Archivist_A1"  → "archivist"
        "ARCHITECT"     → "architect"
        "CODER"         → "coder"

    Args:
        node_name: Raw node identifier (any casing).

    Returns:
        Lower-cased base role string.
    """
    return node_name.lower().split("_")[0]


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_state_mutation(state_update: Dict[str, Any], current_node: str) -> bool:
    """Validate that only authorised roles can set protected keys.

    Three checks are performed in order:

    1. **Immutable ``task`` field** — no node may include a ``task`` key in
       its state update.
    2. **Protected approval keys** — only ``decision_authority`` roles may
       set ``approved_*`` keys.
    3. **Role-exclusive keys** — each key listed in
       ``ROLE_EXCLUSIVE_KEYS`` may only be written by its owning role.

    Args:
        state_update: Dictionary of state changes proposed by the node.
        current_node: Name of the node making the update (e.g. ``"CORRECTOR_C1"``).

    Returns:
        ``True`` if the mutation is allowed.

    Raises:
        ConstitutionalViolation: If any check fails.
    """
    node_role = extract_role_from_node_name(current_node)
    is_decision_authority = node_role in ROLE_CLASSIFICATION["decision_authority"]

    # ------------------------------------------------------------------
    # Check 0: Immutable task field
    # ------------------------------------------------------------------
    if "task" in state_update:
        violation_msg = (
            f"CONSTITUTIONAL VIOLATION: Node '{current_node}' (role: {node_role}) "
            "attempted to mutate the immutable 'task' field. "
            "The 'task' field is 100% read-only and MUST NOT be modified by any node."
        )
        logger.critical(violation_msg)
        raise ConstitutionalViolation(violation_msg)

    # ------------------------------------------------------------------
    # Check 1: Protected approval keys
    # ------------------------------------------------------------------
    protected_keys_in_update = set(state_update.keys()) & set(PROTECTED_APPROVAL_KEYS)

    if protected_keys_in_update and not is_decision_authority:
        violation_msg = (
            f"CONSTITUTIONAL VIOLATION: Node '{current_node}' (role: {node_role}) "
            f"attempted to set protected keys: {protected_keys_in_update}. "
            "Only decision_authority nodes can set these keys."
        )
        logger.critical(violation_msg)
        raise ConstitutionalViolation(violation_msg)

    # ------------------------------------------------------------------
    # Check 2: Role-exclusive keys
    # ------------------------------------------------------------------
    for key in state_update.keys():
        for authorised_role, exclusive_keys in ROLE_EXCLUSIVE_KEYS.items():
            if key in exclusive_keys and node_role != authorised_role:
                violation_msg = (
                    f"CONSTITUTIONAL VIOLATION: Node '{current_node}' (role: {node_role}) "
                    f"attempted to set '{key}' which is exclusive to role '{authorised_role}'."
                )
                logger.critical(violation_msg)
                raise ConstitutionalViolation(violation_msg)

    # ------------------------------------------------------------------
    # All checks passed
    # ------------------------------------------------------------------
    logger.debug("Constitutional validation passed for %s", current_node)
    return True


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def emit_audit_for_violation(
    node_name: str,
    violation: ConstitutionalViolation,
) -> Dict[str, Any]:
    """Create and log an audit entry for a constitutional violation.

    Args:
        node_name: The node that caused the violation.
        violation: The exception instance.

    Returns:
        The audit entry dictionary (for optional downstream use).
    """
    audit_entry: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "severity": "CRITICAL",
        "type": "CONSTITUTIONAL_VIOLATION",
        "node": node_name,
        "violation": str(violation),
    }
    logger.critical("AUDIT: %s", audit_entry)
    return audit_entry


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def enforce_constitutional_rules(node_func):
    """Decorator that enforces constitutional rules on every node execution.

    Applied to **all** node functions by the Graph Controller.  The wrapper:

    1. Executes the node logic.
    2. Validates the proposed state update against constitutional rules.
    3. On violation → emits an audit entry and returns an HIR state update
       so the system halts immediately.

    Usage::

        @enforce_constitutional_rules
        def architect(state: dict) -> dict:
            ...
    """

    @wraps(node_func)
    def wrapper(state: dict) -> dict:
        node_name = node_func.__name__.upper()
        # Keep a shallow copy so we can inspect original state on failure
        _original_state = state.copy()  # noqa: F841 — reserved for error diagnostics

        try:
            # Execute the node
            state_update = node_func(state)

            # Validate the proposed mutation
            validate_state_mutation(
                state_update=state_update,
                current_node=node_name,
            )

            # Validation passed — return the update
            return state_update

        except ConstitutionalViolation as exc:
            logger.critical(
                "CONSTITUTIONAL VIOLATION in %s: %s", node_name, exc
            )
            emit_audit_for_violation(node_name, exc)

            # Immediate HIR — system cannot proceed
            return {
                "status": "HIR",
                "HIR_reason": f"Constitutional violation in {node_name}: {exc!s}",
                "violated_at": node_name,
            }

    return wrapper
