"""Persistence Utility for Orchestral Machine.

Handles saving task execution results (code, state, reports) to the Finish/ folder.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

def save_task_results(task_id: str, state: Dict[str, Any]) -> None:
    """Save generated code, full state snapshot, and reports to a session folder.
    
    Args:
        task_id: Unique identifier for the task session.
        state: The final GraphState dictionary.
    """
    session_dir = Path("Finish") / task_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving task results to {session_dir}")

    # 1. Save Code
    code = state.get("code", {})
    if code:
        for filename, content in code.items():
            filepath = session_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    # 2. Save Full State Snapshot
    snapshot_path = session_dir / "full_state.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

    # 3. Save Archivist Results
    # A1 logs are already in session/artifacts, so we only need to ensure
    # A2 Meta-Summaries (which live in global data/archive) are correctly
    # COPIED into the session record for export/completeness.
    
    archive_dir = session_dir / "data" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Copy A2 Meta-Summary if it exists in state refs
    meta_summary_ref = state.get("archived_meta_summary_ref")
    if meta_summary_ref and isinstance(meta_summary_ref, dict):
        original_path = meta_summary_ref.get("path")
        if original_path:
            try:
                import shutil
                src = Path(original_path)
                if src.exists():
                    dst = archive_dir / src.name
                    shutil.copy2(src, dst)
                    logger.info(f"Copied Meta-Summary to session dir: {dst}")
            except Exception as e:
                logger.warning(f"Failed to copy Meta-Summary to session dir: {e}")

    # Dump the raw dictionaries just in case
    meta_summary = state.get("meta_summary")
    if meta_summary:
         with open(archive_dir / "archivist_a2_meta_summary_raw.json", "w", encoding="utf-8") as f:
            json.dump(meta_summary, f, indent=2, default=str)
            
    logger.info(f"Task results successfully persisted for {task_id}")
