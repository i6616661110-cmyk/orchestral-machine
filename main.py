"""Orchestral Machine — Entry Point.

Provides operating modes:
- `python main.py run`
- `python main.py serve`
- `python main.py serve-bot`
- `python main.py validate-prompt <file_path>`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from src.version import __version__

logger = logging.getLogger(__name__)
MAX_TASK_CHARS = 50_000
DEFAULT_TASK = "Create a Python script that calculates the 100th Fibonacci number."


def setup_logging() -> None:
    """Configure root logging for CLI, API, and dynamic session handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _find_placeholders(text: str) -> list[str]:
    """Find placeholder markers that likely indicate unfinished prompts."""
    markers = {
        "TODO": r"(?i)\btodo\b",
        "STUB": r"(?i)\bstub\b",
    }
    hits: list[str] = []
    for name, pattern in markers.items():
        if re.search(pattern, text):
            hits.append(name)
    return hits


def _validate_prompt_file(file_path: str) -> tuple[str, list[str]]:
    """Validate a prompt file and return (content, warnings)."""
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Prompt file not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Prompt path is not a file: {file_path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"Prompt file is not readable: {file_path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Prompt file must be UTF-8 encoded: {file_path}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read prompt file: {file_path} ({exc})") from exc

    if len(content) > MAX_TASK_CHARS:
        raise ValueError(
            f"Prompt file too long: {len(content)} chars (max {MAX_TASK_CHARS})."
        )

    warnings = [f"Detected placeholder marker: {marker}" for marker in _find_placeholders(content)]
    return content, warnings


def _resolve_task_text(args: argparse.Namespace) -> str:
    """Resolve task text from --task or --task-file with validation."""
    if getattr(args, "task_file", None):
        content, warnings = _validate_prompt_file(args.task_file)
        for warning in warnings:
            logger.warning("Prompt validation warning: %s", warning)
        return content

    task_text = getattr(args, "task", DEFAULT_TASK)
    if len(task_text) > MAX_TASK_CHARS:
        raise ValueError(f"--task is too long: {len(task_text)} chars (max {MAX_TASK_CHARS}).")
    return task_text


def _print_startup_banner() -> None:
    """Print startup banner and key configuration (rich if available)."""
    from src.config import MODEL_MAPPING, RECURSION_LIMIT

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print("=" * 56)
        print(f" Orchestral Machine {__version__} ".center(56, "="))
        print("=" * 56)
        print(f"Recursion limit: {RECURSION_LIMIT}")
        print("Model mappings:")
        for role, model in MODEL_MAPPING.items():
            print(f"  - {role}: {model}")
        return

    console = Console()
    console.print(
        Panel.fit(
            f"[bold cyan]Orchestral Machine {__version__}[/bold cyan]\n"
            "[dim]Deterministic autonomous coding pipeline[/dim]",
            border_style="cyan",
        )
    )
    table = Table(title="Configuration Summary", show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column("Value", overflow="fold")
    table.add_row("Recursion limit", str(RECURSION_LIMIT))
    for role, model in MODEL_MAPPING.items():
        table.add_row(role, model)
    console.print(table)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_code(state: dict) -> None:
    """Print generated code files (truncated to 100 lines each)."""
    code: Dict[str, str] = state.get("code", {})
    if not code:
        print("\n--- No code generated ---")
        return

    for filename, content in code.items():
        lines = content.splitlines()
        truncated = len(lines) > 100
        print(f"\n{'='*60}")
        print(f"FILE: {filename}  ({len(lines)} lines)")
        print("=" * 60)
        for line in lines[:100]:
            print(line)
        if truncated:
            print(f"\n... [{len(lines) - 100} more lines truncated] ...")


def _print_audit_log(state: dict) -> None:
    """Print audit log as a compact table."""
    audit_log: list = state.get("audit_log", [])
    if not audit_log:
        print("\n--- No audit log entries ---")
        return

    print(f"\n{'='*60}")
    print("AUDIT LOG")
    print("=" * 60)
    header = f"{'Node':<18} {'Model':<32} {'Status':<14} {'Timestamp'}"
    print(header)
    print("-" * len(header))

    for entry in audit_log:
        if isinstance(entry, dict):
            node = entry.get("node", "?")
            model = entry.get("model_id", "?")
            status = entry.get("status", "?")
            ts = entry.get("timestamp", "?")
        else:
            node = getattr(entry, "node", "?")
            model = getattr(entry, "model_id", "?")
            status = getattr(entry, "status", "?")
            ts = getattr(entry, "timestamp", "?")
        print(f"{node:<18} {model:<32} {status:<14} {ts}")


def _print_escalation_summary(state: dict) -> None:
    """Print escalation summary as JSON."""
    audit_log: list = state.get("audit_log", [])
    corrector_models: list = []
    for entry in audit_log:
        node = entry.get("node", "") if isinstance(entry, dict) else getattr(entry, "node", "")
        model = entry.get("model_id", "") if isinstance(entry, dict) else getattr(entry, "model_id", "")
        if "CORRECTOR" in node and model and model not in corrector_models:
            corrector_models.append(model)

    summary = {
        "correction_attempt": state.get("correction_attempt", 0),
        "loop_iteration": state.get("loop_iteration", 0),
        "archival_attempt": state.get("archival_attempt", 0),
        "correctors_invoked": corrector_models,
        "final_status": state.get("status"),
    }
    print(f"\n{'='*60}")
    print("ESCALATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


def _display_success(state: dict) -> None:
    """Display results for a successful run."""
    _print_code(state)
    _print_audit_log(state)
    _print_escalation_summary(state)
    ref = state.get("archived_summary_ref")
    if ref:
        print(f"\n{'='*60}")
        print("ARCHIVED SUMMARY REF")
        print("=" * 60)
        if isinstance(ref, dict):
            print(json.dumps(ref, indent=2))
        else:
            print(ref)


def _display_hir(state: dict) -> None:
    """Display results for a HIR (Human Intervention Required) outcome."""
    ref = state.get("archived_summary_ref")
    if ref:
        print(f"\n{'='*60}")
        print("ARCHIVED SUMMARY REF")
        print("=" * 60)
        if isinstance(ref, dict):
            print(json.dumps(ref, indent=2))
        else:
            print(ref)

    print("\n" + "!" * 60)
    print("  ⚠️  HUMAN INTERVENTION REQUIRED (HIR)")
    print("!" * 60)

    event_log: list = state.get("event_log", [])
    if event_log:
        print("\nRecent events:")
        for evt in event_log[-5:]:
            print(f"  - {json.dumps(evt)}" if isinstance(evt, dict) else f"  - {evt}")

    error_logs: list = state.get("error_logs", [])
    if error_logs:
        print("\nRecent errors:")
        for err in error_logs[-5:]:
            print(f"  - {json.dumps(err)}" if isinstance(err, dict) else f"  - {err}")


def _save_full_state(state: dict, task_id: str) -> None:
    """Save the complete state dict to a JSON snapshot."""
    snapshots_dir = Path("./snapshots")
    snapshots_dir.mkdir(exist_ok=True)
    filepath = snapshots_dir / f"state_{task_id}.json"
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str)
    print(f"\nState snapshot saved: {filepath}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the graph with a task and display results."""
    from src.execution_engine import run_task_simple
    from src.integrations.persistence import save_task_results
    from src.tools.view_logs import print_session_logs

    try:
        task_text = _resolve_task_text(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    task_id = uuid.uuid4().hex
    result: dict = {"status": "ERROR", "error": "Execution did not complete"}
    exit_code = 0
    try:
        logger.info("Invoking streaming execution (task_id=%s)", task_id)
        result = run_task_simple(task_text, task_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph execution failed")
        result = {"status": "ERROR", "error": str(exc)}
        exit_code = 2
    finally:
        print("\n=== AUTOMATIC LOG VIEW ===\n")
        try:
            print_session_logs(Path("Finish"), task_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to run automatic log viewer for task_id=%s", task_id)

    if not isinstance(result, dict):
        result = result.model_dump() if hasattr(result, "model_dump") else dict(result)

    final_status = result.get("status")
    if final_status == "HIR":
        _display_hir(result)
        exit_code = max(exit_code, 1)
    elif final_status == "ERROR":
        print(f"\nExecution failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
        exit_code = max(exit_code, 2)
    else:
        _display_success(result)

    save_task_results(task_id, result)

    if args.save_full_state:
        _save_full_state(result, task_id)

    if exit_code != 0:
        sys.exit(exit_code)


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI HTTP server."""
    import uvicorn
    from fastapi import FastAPI

    from src.api.control_endpoints import router

    fastapi_app = FastAPI(title="Orchestral Machine", version=__version__)
    fastapi_app.include_router(router)

    logger.info("Starting API server on %s:%d", args.host, args.port)
    uvicorn.run(fastapi_app, host=args.host, port=args.port)


def _cmd_serve_bot(args: argparse.Namespace) -> None:
    """Start the Telegram bot listener."""
    from src.integrations.telegram_listener import run_listener

    run_listener()


def _cmd_validate_prompt(args: argparse.Namespace) -> None:
    """Validate prompt file syntax and basic quality checks."""
    try:
        content, warnings = _validate_prompt_file(args.file_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Prompt file is valid UTF-8: {args.file_path}")
    print(f"Length: {len(content)} chars (max {MAX_TASK_CHARS})")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No placeholder warnings detected.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI."""
    parser = argparse.ArgumentParser(
        prog="orchestral-machine",
        description="Orchestral Machine — autonomous coding factory.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Execute the graph pipeline.")
    run_parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task description for the graph (default: Fibonacci demo).",
    )
    run_parser.add_argument(
        "--task-file",
        default=None,
        help="Path to a UTF-8 prompt file. If set, it is used instead of --task.",
    )
    run_parser.add_argument(
        "--save-full-state",
        action="store_true",
        default=False,
        help="Save the complete final state to ./snapshots/.",
    )

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI control server.")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")

    subparsers.add_parser("serve-bot", help="Start the Telegram bot listener.")

    validate_prompt_parser = subparsers.add_parser(
        "validate-prompt",
        help="Validate a prompt file (UTF-8, length, placeholders).",
    )
    validate_prompt_parser.add_argument("file_path", help="Path to prompt markdown/text file.")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    setup_logging()
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()

    command = args.command
    if command is None:
        args.task = DEFAULT_TASK
        args.task_file = None
        args.save_full_state = False
        command = "run"

    if command in {"run", "serve", "serve-bot"}:
        _print_startup_banner()

    if command in {"run", "serve-bot"}:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY is not set. Add it to .env or export it.", file=sys.stderr)
            sys.exit(1)

    if command == "run":
        _cmd_run(args)
    elif command == "serve":
        _cmd_serve(args)
    elif command == "serve-bot":
        _cmd_serve_bot(args)
    elif command == "validate-prompt":
        _cmd_validate_prompt(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
