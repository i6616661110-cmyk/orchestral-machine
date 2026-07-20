"""Orchestral Machine — Operator Control CLI.

Click-based command-line interface for operator control of the Orchestral
Machine LangGraph state machine.  This module is a **thin delegation layer**
— all business logic lives in ``src/control_interface.py``.

Usage::

    python control_cli.py <command> [options]

Commands::

    halt              Halt the system and save a checkpoint.
    resume            Resume execution from a checkpoint.
    force-reset       Confirmed hard reset with archival snapshot.
    status            Display current system status.
    checkpoints       Checkpoint management (list, inspect, validate, clean).
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from typing import Optional

import click

from src.checkpoint import (
    CheckpointCorruptedError,
    get_latest_checkpoint,
    load_checkpoint,
    validate_checkpoint,
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


# ---------------------------------------------------------------------------
# State Access Helper
# ---------------------------------------------------------------------------


def _get_current_state() -> dict:
    """Load current graph state for CLI operations.

    In a running system this would connect to the active graph runtime.
    For CLI operations (HALT, RESUME, FORCE_RESET) the state is typically
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
# Main CLI Group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Orchestral Machine — Operator Control CLI."""


# ---------------------------------------------------------------------------
# halt
# ---------------------------------------------------------------------------


@cli.command()
def halt() -> None:
    """Halt the system and save a checkpoint."""
    try:
        state = _get_current_state()
        result = operator_halt(state)
        click.secho("✓ System halted successfully.", fg="green")
        click.echo(f"  Status     : {result['status']}")
        click.echo(f"  Checkpoint : {result['checkpoint']}")
        click.echo(f"  Message    : {result['message']}")
    except Exception as exc:
        _handle_unexpected(exc)


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(exists=False),
    help="Path to a specific checkpoint file. Defaults to latest.",
)
@click.option(
    "--revert-to",
    "revert_to",
    default=None,
    type=click.Choice(
        ["ARCHITECT", "CODER", "REVIEWER", "TESTER", "CORRECTOR", "VERIFIER"],
        case_sensitive=True,
    ),
    help="Rollback state to this node and resume from it.",
)
def resume(checkpoint: Optional[str], revert_to: Optional[str]) -> None:
    """Resume execution from a checkpoint."""
    from src.graph import _controller_metadata

    try:
        # 1. Load checkpoint
        path = checkpoint or get_latest_checkpoint()
        if path is None:
            click.secho("Error: No checkpoint found.", fg="red")
            sys.exit(1)

        checkpoint_data = load_checkpoint(path)
        if not validate_checkpoint(checkpoint_data):
            click.secho(f"Checkpoint corrupted: {path}", fg="red")
            sys.exit(1)

        state = checkpoint_data.get("graph_state", {})
        metadata = checkpoint_data.get("checkpoint_metadata", {})
        task_id = state.get("task_id", "")
        task_text = state.get("task", "")

        if not task_id or not task_text:
            click.secho("Error: Checkpoint has no task_id or task.", fg="red")
            sys.exit(1)

        # 2. Apply rollback if requested
        if revert_to:
            from src.checkpoint import rollback_state_to_node

            state = rollback_state_to_node(state, revert_to)
            state["resume_target"] = revert_to
            click.echo(f"  Rollback to : {revert_to}")
        else:
            # Default: resume at the next scheduled node from checkpoint
            next_node = metadata.get("next_scheduled_node", "ARCHITECT")
            state["resume_target"] = next_node
            click.echo(f"  Resume at   : {next_node}")

        # 3. Override in-memory controller state
        _controller_metadata["system_state"] = "RUNNING"

        click.secho(f"Resuming task {task_id} from {path}", fg="green")

        # 4. Run execution engine with restored state
        from src.execution_engine import run_task_generator
        from src.integrations.persistence import save_task_results

        final_state = None
        for event in run_task_generator(task_text, task_id, resume_state=state):
            event_type = event.get("type")
            if event_type == "STATE":
                role = event.get("role", "?")
                status = event.get("status", "?")
                click.echo(f"  [{role}] {status}")
            elif event_type == "RESULT":
                final_state = event.get("payload", {})
                click.secho("Graph execution completed.", fg="green")
            elif event_type == "ERROR":
                click.secho(f"Error: {event.get('error')}", fg="red")
                sys.exit(1)

        if final_state:
            save_task_results(task_id, final_state)
            click.echo(f"  Final status: {final_state.get('status')}")

    except FileNotFoundError as exc:
        click.secho(f"Error: {exc}", fg="red")
        sys.exit(1)
    except CheckpointCorruptedError as exc:
        click.secho(f"Checkpoint corrupted: {exc}", fg="red")
        sys.exit(1)
    except Exception as exc:
        _handle_unexpected(exc)


# ---------------------------------------------------------------------------
# force-reset
# ---------------------------------------------------------------------------


@cli.command("force-reset")
@click.option(
    "--confirm",
    "confirmed",
    is_flag=True,
    default=False,
    help="Skip interactive confirmation prompt.",
)
def force_reset(confirmed: bool) -> None:
    """Confirmed hard reset with archival snapshot."""
    try:
        state = _get_current_state()
        result = operator_force_reset(state, confirmation=confirmed)

        if result["status"] == "CONFIRMATION_REQUIRED":
            click.secho("FORCE_RESET requires confirmation.", fg="yellow")
            click.echo("\nCurrent state summary:")
            summary = result.get("current_state_summary", {})
            for key, value in summary.items():
                click.echo(f"  {key}: {value}")
            click.echo()

            if click.confirm("Are you sure you want to reset?"):
                result = operator_force_reset(state, confirmation=True)
            else:
                click.secho("Reset cancelled.", fg="yellow")
                return

        if result["status"] == "RESET_DENIED":
            click.secho(f"Reset denied: {result['message']}", fg="red")
            sys.exit(1)

        if result["status"] == "RESET_COMPLETE":
            click.secho("✓ Force reset complete.", fg="green")
            click.echo(f"  Snapshot ref       : {result['snapshot_ref']}")
            click.echo(f"  New loop iteration : {result['new_loop_iteration']}")

    except InvalidStateError as exc:
        click.secho(f"Error: {exc}", fg="red")
        sys.exit(1)
    except Exception as exc:
        _handle_unexpected(exc)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@cli.command()
def status() -> None:
    """Display current system status."""
    try:
        state = _get_current_state()
        result = get_system_status(state)
        click.secho("System Status", fg="green", bold=True)
        for key, value in result.items():
            click.echo(f"  {key}: {value}")
    except Exception as exc:
        _handle_unexpected(exc)


# ---------------------------------------------------------------------------
# checkpoints (command group)
# ---------------------------------------------------------------------------


@cli.group("checkpoints")
def checkpoints_group() -> None:
    """Checkpoint management commands."""


@checkpoints_group.command("list")
def checkpoints_list_cmd() -> None:
    """List all available checkpoints."""
    try:
        paths = checkpoint_list()
        if not paths:
            click.secho("No checkpoints found.", fg="yellow")
            return
        click.secho(f"Found {len(paths)} checkpoint(s):", fg="green")
        for path in paths:
            click.echo(f"  {path}")
    except Exception as exc:
        _handle_unexpected(exc)


@checkpoints_group.command("inspect")
@click.argument("checkpoint_file")
def checkpoints_inspect_cmd(checkpoint_file: str) -> None:
    """Inspect a specific checkpoint file."""
    try:
        data = checkpoint_inspect(checkpoint_file)
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
    except FileNotFoundError as exc:
        click.secho(f"Error: {exc}", fg="red")
        sys.exit(1)
    except Exception as exc:
        _handle_unexpected(exc)


@checkpoints_group.command("validate")
@click.argument("checkpoint_file")
def checkpoints_validate_cmd(checkpoint_file: str) -> None:
    """Validate checkpoint integrity."""
    try:
        result = checkpoint_validate(checkpoint_file)
        if result["valid"]:
            click.secho("✓ Valid", fg="green")
        else:
            click.secho("✗ Invalid", fg="red")
            sys.exit(1)
    except FileNotFoundError as exc:
        click.secho(f"Error: {exc}", fg="red")
        sys.exit(1)
    except Exception as exc:
        _handle_unexpected(exc)


@checkpoints_group.command("clean")
@click.option(
    "--keep",
    default=10,
    type=int,
    show_default=True,
    help="Number of most recent checkpoints to retain.",
)
def checkpoints_clean_cmd(keep: int) -> None:
    """Clean old checkpoints, keeping the most recent N."""
    try:
        result = checkpoint_clean(keep=keep)
        click.secho("✓ Checkpoint cleanup complete.", fg="green")
        click.echo(f"  Deleted : {result['deleted']}")
        click.echo(f"  Kept    : {result['kept']}")
    except Exception as exc:
        _handle_unexpected(exc)


# ---------------------------------------------------------------------------
# Shared Error Handler
# ---------------------------------------------------------------------------


def _handle_unexpected(exc: Exception) -> None:
    """Handle unexpected exceptions with traceback and exit code 2."""
    click.secho(f"Unexpected error: {exc}", fg="red")
    click.echo(traceback.format_exc())
    sys.exit(2)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
