"""Orchestral Machine — REST API Control Endpoints.

FastAPI router providing HTTP access to operator control commands, system
status, and checkpoint management.  This module is a **thin delegation
layer** — all business logic lives in ``src/control_interface.py``.

Endpoints::

    POST /api/control                 Primary operator control (HALT / RESUME / FORCE_RESET)
    GET  /api/status                  Current system status
    GET  /api/checkpoints             List available checkpoints
    GET  /api/checkpoints/{path}      Inspect a specific checkpoint
    POST /api/checkpoints/validate    Validate a checkpoint file
    POST /api/checkpoints/clean       Clean old checkpoints
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.checkpoint import (
    CheckpointCorruptedError,
    get_latest_checkpoint,
    load_checkpoint,
)
from src.control_interface import (
    InvalidStateError,
    checkpoint_clean,
    checkpoint_inspect,
    checkpoint_list,
    checkpoint_validate,
    get_system_status,
    operator_force_reset,
    operator_halt,
    operator_resume,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["control"])


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ControlRequest(BaseModel):
    """Primary control endpoint request body."""

    command: str = Field(
        ...,
        pattern="^(HALT|RESUME|FORCE_RESET)$",
        description="Operator command: HALT, RESUME, or FORCE_RESET.",
    )
    confirmation: bool = False
    checkpoint: Optional[str] = None


class ValidateRequest(BaseModel):
    """Checkpoint validation request body."""

    path: str


class CleanRequest(BaseModel):
    """Checkpoint clean request body."""

    keep: int = 10


# ---------------------------------------------------------------------------
# State Access Helper
# ---------------------------------------------------------------------------

def _get_current_state() -> dict:
    """Load current graph state for API operations.

    In a running system this would connect to the active graph runtime.
    For API operations (HALT, RESUME, FORCE_RESET) the state is typically
    loaded from the latest checkpoint or initialised as empty.

    Returns:
        Flat GraphState dict, either from the latest checkpoint or a
        minimal empty default.
    """
    try:
        latest = get_latest_checkpoint()
        if latest is not None:
            checkpoint = load_checkpoint(latest)
            state = checkpoint.get("graph_state", {})
            if state:
                logger.info("State loaded from checkpoint: %s", latest)
                return state
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load state from checkpoint: %s", exc)

    # Minimal empty state with required defaults
    return {
        "task": "",
        "task_id": "",
        "plan": None,
        "code": {},
        "status": None,
        "loop_iteration": 0,
        "correction_attempt": 0,
        "archival_attempt": 0,
        "verifier_reset_used": False,
        "system_flags": {},
        "error_logs": [],
        "event_log": [],
        "audit_log": [],
    }


# ---------------------------------------------------------------------------
# Security Dependency
# ---------------------------------------------------------------------------
import os
from fastapi import Header, Depends

def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify that the X-API-Key header matches the configured secret."""
    control_key = os.getenv("CONTROL_API_KEY")
    if not control_key:
        # Fail secure if not configured
        logger.error("CONTROL_API_KEY is not configured in environment.")
        raise HTTPException(status_code=500, detail="Server security misconfiguration.")
    
    if x_api_key != control_key:
        logger.warning("Invalid API Key attempt.")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


# ---------------------------------------------------------------------------
# Primary Control Endpoint
# ---------------------------------------------------------------------------

@router.post("/control", dependencies=[Depends(verify_api_key)])
def control(request: ControlRequest) -> dict:
    """Execute an operator control command (HALT, RESUME, FORCE_RESET)."""
    state = _get_current_state()

    try:
        if request.command == "HALT":
            return operator_halt(state)

        if request.command == "RESUME":
            return operator_resume(state, checkpoint_path=request.checkpoint)

        # FORCE_RESET
        return operator_force_reset(state, confirmation=request.confirmation)

    except InvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CheckpointCorruptedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /api/control")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# System Status
# ---------------------------------------------------------------------------

@router.get("/status")
def status() -> dict:
    """Return current system status."""
    try:
        state = _get_current_state()
        return get_system_status(state)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /api/status")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------

@router.get("/checkpoints")
def checkpoints_list_endpoint() -> dict:
    """List all available checkpoints."""
    try:
        paths = checkpoint_list()
        return {"checkpoints": paths}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /api/checkpoints")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/checkpoints/{checkpoint_file:path}")
def checkpoints_inspect_endpoint(checkpoint_file: str) -> dict:
    """Inspect a specific checkpoint file."""
    try:
        return checkpoint_inspect(checkpoint_file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /api/checkpoints/%s", checkpoint_file)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/checkpoints/validate", dependencies=[Depends(verify_api_key)])
def checkpoints_validate_endpoint(request: ValidateRequest) -> dict:
    """Validate a checkpoint file."""
    try:
        result = checkpoint_validate(request.path)
        return {"valid": result.get("valid", False), "path": request.path}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /api/checkpoints/validate")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/checkpoints/clean", dependencies=[Depends(verify_api_key)])
def checkpoints_clean_endpoint(request: CleanRequest) -> dict:
    """Remove old checkpoints, keeping the most recent N."""
    try:
        result = checkpoint_clean(keep=request.keep)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /api/checkpoints/clean")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
