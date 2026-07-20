"""Orchestral Machine -- Archivist_A2 Node.

Root Cause & Meta-Summary Analyst phase of archival flow.
Validates A1 summaries, performs cross-record root cause analysis,
and persists long-term meta-summaries under
``data/archive/meta_summaries/``.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import MODEL_MAPPING, NODE_TIMEOUT_LONG
from src.enforcement import enforce_constitutional_rules
from src.llm_factory import build_chat_llm, invoke_node_llm
from src.nodes.base import BaseNode
from src.schemas import ArchivistA2Output

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT_A2_TEMPLATE = """\
SYSTEM: You are ARCHIVIST_A2 role in the Orchestral Machine. Act as Senior Historian, Root Cause and Meta-Summary Analyst performing quality summarization. Your task is to receive batches of new ARCHIVIST_A1 summaries, produce root cause analyses and validate factual accuracy, and create high-level "Meta-Summaries".

CONTEXT:
- You receive batch of new summaries from Archivist_A1 (not the entire DB).
  - optional previous meta summaries for context
- Your role is to verify accuracy and produce high-quality root cause analysis.

YOUR_MANDATE:
1. **Validation:**
   - Retrieve new A1 summaries (JSON), perform a consistency check on the incoming A1 batch and accessible snapshots. Ensure timestamps align and error traces are complete before synthesis. If data is corrupt, return status='archived_summary_corrupt'.
   - Cross-reference new summaries with existing database records to check for contradictions or duplicates.
2. **Root Cause:** Analyze patterns to determine WHY the architecture or code failed. Identify root causes across multiple failure records.
3. **Synthesis:** Produce 'archived_meta_summary' JSON containing with actionable insights for Architect:
   - Upsert any derived summary documents or metadata to Qdrant, and produce state['archived_meta_summary_ref'] = {{"id": (uuid), "created_at", "consolidated_session_summary", "verified_failure_facts", "root_causes"(with evidence pointers to "chunk_ids"), "actionable_metrics", "meta_file": "data/archived_meta_summary_<id>.json"}}.
   - Ensure no distortion of failure causes. Emphasize factual accuracy and avoid editorialization.

STRICT_PROHIBITIONS:
- You MUST NOT modify raw A1 records.
- You MUST NOT speculate beyond evidence.
- You MUST NOT recommend specific solutions (only identify patterns).
- You MUST NOT reindex entire DB (use A1 outputs only).

OUTPUT_FORMAT:
Return a JSON object matching the ARCHIVIST_A2 schema. Strict JSON only.
{schema}

Create meta-summary document with:
- Verified failure patterns
- Root cause analysis
- Statistical summary of attempt distributions

- AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the global state yourself. The Graph Controller will append this record."""


@enforce_constitutional_rules
def archivist_a2_node(state: dict) -> dict:
    """Archivist A2 node -- Root Cause & Meta-Summary Analyst.

    Validates A1 factual accuracy, performs root cause analysis across
    failure records, and produces high-level meta-summaries with
    actionable insights for the Architect.

    Args:
        state: Full graph state dictionary.

    Returns:
        Flat dict with ``archived_meta_summary_ref``, ``meta_summary``,
        ``status``, and audit metadata for the Graph Controller to merge
        into state.
    """
    archived_summary_ref = state.get("archived_summary_ref")
    archivist_feedback = state.get("archivist_feedback")
    task_id: str = state["task_id"]

    summary_content: str | None = None
    if isinstance(archived_summary_ref, dict) and "path" in archived_summary_ref:
        try:
            summary_content = Path(archived_summary_ref["path"]).read_text(
                encoding="utf-8"
            )
        except (OSError, IOError) as exc:
            logger.warning(
                "Failed to read A1 summary file at %s: %s",
                archived_summary_ref["path"],
                exc,
            )

    node = BaseNode(ArchivistA2Output, _SYSTEM_PROMPT_A2_TEMPLATE)
    seed = node.generate_seed(task_id)
    system_prompt = node.build_system_prompt()

    human_parts: list[str] = []

    if archived_summary_ref is not None:
        human_parts.append(
            f"Archived Summary Reference (A1 summaries to process):\n"
            f"{json.dumps(archived_summary_ref, indent=2) if isinstance(archived_summary_ref, dict) else str(archived_summary_ref)}"
        )
        if summary_content is not None:
            human_parts.append(f"A1 Summary File Content:\n{summary_content}")
        else:
            human_parts.append(
                "A1 Summary File Content: NOT AVAILABLE — file read failed or path missing."
            )
    else:
        human_parts.append(
            "Archived Summary Reference: None (no A1 summaries available)"
        )

    if archivist_feedback is not None:
        human_parts.append(
            f"\nArchivist Feedback (correction instructions):\n"
            f"{json.dumps(archivist_feedback, indent=2) if isinstance(archivist_feedback, (dict, list)) else str(archivist_feedback)}"
        )

    human_parts.append(f"\nTask ID: {task_id}")
    human_parts.append(f"Seed: {seed}")

    human_content = "\n\n".join(human_parts)

    model_id = MODEL_MAPPING["ARCHIVIST_A2"]
    llm = build_chat_llm(
        model_id=model_id,
        timeout=NODE_TIMEOUT_LONG,
        seed=int(seed, 16) % (2**31),
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    parsed_output = invoke_node_llm(
        llm=llm,
        model_id=model_id,
        messages=messages,
        output_schema_cls=ArchivistA2Output,
        seed=seed,
        logger=logger,
    )

    if parsed_output is None:
        logger.error("Archivist A2 node exhausted all attempts.")
        now = datetime.utcnow().isoformat()
        return {
            "archived_meta_summary_ref": None,
            "meta_summary": None,
            "status": "FAIL",
            "error_logs": state.get("error_logs", [])
            + [
                {
                    "node": "ARCHIVIST_A2",
                    "error": "Failed to generate valid ArchivistA2Output after retries.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "archivist_a2_failed",
                    "detail": "Failed to generate valid ArchivistA2Output after retries.",
                    "timestamp": now,
                }
            ],
            "active_model": model_id,
            "updated_at": now,
        }

    now = datetime.utcnow().isoformat()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")

    archive_dir = Path("data") / "archive" / "meta_summaries"
    archive_dir.mkdir(parents=True, exist_ok=True)
    meta_summary_path = (archive_dir / f"meta_summary_{task_id}_{ts}.json").resolve()
    meta_summary_path.write_text(
        parsed_output.model_dump_json(indent=2), encoding="utf-8"
    )

    new_meta_summary_ref = {
        "id": parsed_output.meta_summary_id,
        "created_at": now,
        "source_records": parsed_output.source_records,
        "root_causes_count": len(parsed_output.root_causes),
        "failure_patterns_count": len(parsed_output.failure_patterns),
        "status": parsed_output.status,
        "path": str(meta_summary_path),
    }

    meta_summary = {
        "meta_summary_id": parsed_output.meta_summary_id,
        "correction_instructions": parsed_output.correction_instructions,
        "root_causes": [rc.model_dump() for rc in parsed_output.root_causes],
        "failure_patterns": [fp.model_dump() for fp in parsed_output.failure_patterns],
        "recommendations_for_architect": parsed_output.recommendations_for_architect,
        "statistics": parsed_output.statistics.model_dump()
        if parsed_output.statistics
        else None,
    }

    if parsed_output.status == "archived_meta_summary":
        event_name = "archivist_a2_meta_summary"
        event_detail = (
            f"Archivist A2 produced meta-summary: "
            f"{len(parsed_output.root_causes)} root cause(s), "
            f"{len(parsed_output.failure_patterns)} pattern(s)"
        )
    elif parsed_output.status == "archived_summary_corrupt":
        event_name = "archivist_a2_corrupt"
        event_detail = "Archivist A2 detected corrupt A1 summary data"
    else:
        event_name = "archivist_a2_error"
        event_detail = f"Archivist A2 meta-summary error: {parsed_output.status}"

    return {
        "archived_meta_summary_ref": new_meta_summary_ref,
        "meta_summary": meta_summary,
        "status": parsed_output.status,
        "notes": parsed_output.notes,
        "audit": parsed_output.audit.model_dump() if parsed_output.audit else {},
        "error_logs": state.get("error_logs", []),
        "event_log": state.get("event_log", [])
        + [
            {
                "event": event_name,
                "detail": event_detail,
                "timestamp": now,
            }
        ],
        "active_model": model_id,
        "updated_at": now,
    }
