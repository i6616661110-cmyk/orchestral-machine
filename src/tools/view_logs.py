#!/usr/bin/env python3
"""Log Viewer for Orchestral Machine.

Usage:
    python3 src/tools/view_logs.py [task_id]

If task_id is not provided, it lists available sessions in Finish/.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import TextIO


def _resolve_session_dir(finish_dir: Path, task_id: str) -> Path | None:
    """Resolve task_id to a concrete session directory (supports prefix match)."""
    session_dir = finish_dir / task_id
    if session_dir.exists():
        return session_dir

    matches = [m for m in finish_dir.glob(f"{task_id}*") if m.is_dir()]
    if len(matches) == 1:
        return matches[0]
    return None


def list_sessions(finish_dir: Path, out: TextIO = sys.stdout) -> None:
    """List available session directories in Finish/."""
    print(f"Sessions in {finish_dir}:", file=out)
    if not finish_dir.exists():
        print("  (No Finish/ directory found)", file=out)
        return

    sessions = []
    for item in finish_dir.iterdir():
        if item.is_dir() and item.name != "UNKNOWN":
            sessions.append(item.name)

    for sess in sorted(sessions, reverse=True):
        print(f"  - {sess}", file=out)


def print_session_logs(finish_dir: Path | str, task_id: str, out: TextIO = sys.stdout) -> bool:
    """Print API history for one session and return True if logs were shown."""
    finish_path = Path(finish_dir)
    session_dir = _resolve_session_dir(finish_path, task_id)
    if session_dir is None:
        print(f"Session '{task_id}' not found in {finish_path}", file=out)
        return False

    log_file = session_dir / "api_history.jsonl"
    if not log_file.exists():
        print(f"No api_history.jsonl found in {session_dir}", file=out)
        return False

    print(f"Viewing API logs for: {session_dir.name}", file=out)
    print("=" * 60, file=out)

    shown = False
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                shown = True
                try:
                    record = json.loads(line)
                    print(f"\n--- Record {i} ---", file=out)
                    print(f"Time:   {record.get('timestamp')}", file=out)
                    print(f"Node:   {record.get('node')}", file=out)
                    print(f"Model:  {record.get('model')}", file=out)
                    print(f"Attempt:{record.get('attempt')}", file=out)

                    usage = record.get("usage")
                    if usage:
                        print(f"Usage:  {usage}", file=out)

                    outputs = record.get("outputs")
                    if outputs:
                        if isinstance(outputs, dict):
                            print("Output (JSON):", file=out)
                            print(json.dumps(outputs, indent=2), file=out)
                        else:
                            output_text = str(outputs)
                            suffix = "..." if len(output_text) > 500 else ""
                            print(f"Output: {output_text[:500]}{suffix}", file=out)
                    else:
                        print("Output: (None)", file=out)
                except json.JSONDecodeError:
                    print(f"--- Record {i} (Invalid JSON) ---", file=out)
                    print(line.strip()[:200], file=out)
    except Exception as exc:  # noqa: BLE001
        print(f"Error reading log: {exc}", file=out)
        return False

    if not shown:
        print("(No API records found)", file=out)
    return True


def view_session(finish_dir: Path, task_id: str, out: TextIO = sys.stdout) -> None:
    """Backward-compatible wrapper used by CLI mode."""
    print_session_logs(finish_dir, task_id, out=out)


def _discover_finish_dir() -> Path:
    """Locate Finish directory from CWD or repository root."""
    finish_dir = Path.cwd() / "Finish"
    if finish_dir.exists():
        return finish_dir
    return Path(__file__).resolve().parents[2] / "Finish"


def main() -> None:
    """CLI entrypoint for listing or viewing sessions."""
    parser = argparse.ArgumentParser(description="View Orchestral Machine API logs.")
    parser.add_argument("task_id", nargs="?", help="Task ID or prefix to view.")
    args = parser.parse_args()

    finish_dir = _discover_finish_dir()
    if args.task_id:
        view_session(finish_dir, args.task_id)
    else:
        list_sessions(finish_dir)


if __name__ == "__main__":
    main()
