"""Orchestral Machine -- Archivist_A1 Node.

Factual System Historian phase of archival flow.
Collects operational logs/events into structured factual summaries,
chunks and embeds data for vector storage, and persists session-level
artifacts under ``Finish/<task_id>/artifacts/``.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import MODEL_MAPPING, NODE_TIMEOUT_DEFAULT
from src.enforcement import enforce_constitutional_rules
from src.llm_factory import build_chat_llm, invoke_node_llm
from src.nodes.base import BaseNode
from src.schemas import ArchivistA1Output

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT_A1_TEMPLATE = """\
SYSTEM: You are ARCHIVIST_A1 role in the Orchestral Machine. Act as a Factual System Historian performing archival operations. Your task is to collect operational (raw) logs and events into structured summaries to produce provisional factual summaries for vector storage (Qdrant).

CONTEXT:
- You receive state['error_logs'], session snapshot, and failure history.
	- state['archivist_queue'] (raw logs and session snapshot)
	- state['archival_attempt']
- Your role is to create accurate, factual records for future reference.
- Check if `state['archivist_feedback']` exists (feedback from A2 or Validator). If so, fix the specific issues mentioned before archiving again.

YOUR_MANDATE:
1. **Facts Only:** Extract factual information from error logs and session state.
   - Create structured summaries FACTS (what happened) from HYPOTHESES (why it happened). Do not speculate.
2. **Chunking & Embeddings:** Chunk and embed data into semantic pieces (<= 2000 tokens per chunk) for vector storage. Group related errors.
  - For each chunk compute embeddings and upsert to Qdrant collection "orchestral_archive"; ensure metadata includes task_id, source_node, timestamp.
3. **Persistence:** Ensure data is successfully upserted to Qdrant before returning the ID.
   - Always complete the upsert; request increment state['archival_attempt'] in your JSON output.
   - For each failure mode, produce a provisional archived_summary JSON containing: facts, failure traces, references to full snapshots, and chunk_ids. Include a "code_context" field. This MUST be an exact, read-only excerpt from state['code'] relevant to the error (max 20 lines). Mask any secrets/keys. This is a factual citation, not code generation.
   - In addition to chunk-level vectors, persist a snapshot metadata object that references the full session snapshot (path to `snapshots/session_<id>.json`) which stores the full plan, full code bundle, test_outputs, and environment metadata. Archivist_A1 MUST NOT duplicate full code text into vector embeddings, but MUST record snapshot references for retrieval by Archivist_A2 and Architect.
     - Persist the archived_summary JSON to local storage (e.g., data/archived_summary_<id>.json) and return state['archived_summary_ref'] = {{"id": (uuid), "created_at", "chunk_ids[]", "summaries": {{session_summary, failure_facts, hypotheses, metrics_and_actions}} "summary_file": "data/archived_summary_<id>.json", "chunks": N}}.

STRICT_PROHIBITIONS:
- You MUST NOT editorialize or interpret creatively.
- You MUST NOT bias future decisions with opinions.
- You MUST NOT alter the meaning of error messages or outcomes.
- You MUST NOT perform high-level root cause analysis (A1 is factual).

OUTPUT_FORMAT:
Return a JSON object matching the ARCHIVIST_A1 schema. Strict JSON only.
{schema}

DO NOT return the global state object.
Create vector DB entries with:
- Factual failure descriptions
- Error classifications
- Code snippets that failed
- Timestamps and attempt numbers
- AUDIT REQUIREMENT: Include an `audit` field in your JSON output. Do NOT mutate the global state yourself. The Graph Controller will append this record.

**State Mutation:** You cannot change `state['archival_attempt']` directly. Instead, include "increment_archival_attempt": true in your output JSON.
The Graph Controller will perform the increment atomically and apply these values to the global state."""


@enforce_constitutional_rules
def archivist_a1_node(state: dict) -> dict:
    """Archivist A1 node -- Factual System Historian.

    Collects operational logs and events, creates structured factual
    summaries, chunks and embeds data for Qdrant vector storage.

    Args:
        state: Full graph state dictionary.

    Returns:
        Flat dict with ``archived_summary_ref``, ``status``, and audit
        metadata for the Graph Controller to merge into state.
    """
    archivist_queue: dict | list | None = state.get("archivist_queue")
    error_logs: list = state.get("error_logs", [])
    archived_summary_ref = state.get("archived_summary_ref")
    archivist_feedback = state.get("archivist_feedback")
    archival_attempt: int = state.get("archival_attempt", 0)
    task_id: str = state["task_id"]

    node = BaseNode(ArchivistA1Output, _SYSTEM_PROMPT_A1_TEMPLATE)
    seed = node.generate_seed(task_id)
    system_prompt = node.build_system_prompt()

    human_parts: list[str] = []

    if archivist_queue is not None:
        human_parts.append(
            f"Archivist Queue (raw logs and session snapshot):\n"
            f"{json.dumps(archivist_queue, indent=2) if isinstance(archivist_queue, (dict, list)) else str(archivist_queue)}"
        )

    human_parts.append(f"\nError Logs:\n{json.dumps(error_logs, indent=2)}")

    human_parts.append(f"\nArchival Attempt: {archival_attempt}")

    if archivist_feedback is not None:
        human_parts.append(
            f"\nArchivist Feedback (correction instructions from Validator/A2):\n"
            f"{json.dumps(archivist_feedback, indent=2) if isinstance(archivist_feedback, (dict, list)) else str(archivist_feedback)}"
        )

    if archived_summary_ref is not None:
        human_parts.append(
            f"\nPrevious Archived Summary Ref:\n"
            f"{json.dumps(archived_summary_ref, indent=2) if isinstance(archived_summary_ref, dict) else str(archived_summary_ref)}"
        )

    human_parts.append(f"\nTask ID: {task_id}")
    human_parts.append(f"Seed: {seed}")

    human_content = "\n\n".join(human_parts)

    model_id = MODEL_MAPPING["ARCHIVIST_A1"]
    llm = build_chat_llm(
        model_id=model_id,
        timeout=NODE_TIMEOUT_DEFAULT,
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
        output_schema_cls=ArchivistA1Output,
        seed=seed,
        logger=logger,
    )

    if parsed_output is None:
        logger.error("Archivist A1 node exhausted all attempts.")
        now = datetime.utcnow().isoformat()
        return {
            "archived_summary_ref": archived_summary_ref,
            "status": "FAIL",
            "error_logs": error_logs
            + [
                {
                    "node": "ARCHIVIST_A1",
                    "error": "Failed to generate valid ArchivistA1Output after retries.",
                    "timestamp": now,
                }
            ],
            "event_log": state.get("event_log", [])
            + [
                {
                    "event": "archivist_a1_failed",
                    "detail": "Failed to generate valid ArchivistA1Output after retries.",
                    "timestamp": now,
                }
            ],
            "active_model": model_id,
            "updated_at": now,
        }

    now = datetime.utcnow().isoformat()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")

    archive_dir = Path("Finish") / task_id / "artifacts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (archive_dir / f"summary_{ts}.json").resolve()
    summary_path.write_text(parsed_output.model_dump_json(indent=2), encoding="utf-8")

    new_archived_summary_ref = {
        "id": parsed_output.records_created[0].record_id
        if parsed_output.records_created
        else None,
        "created_at": now,
        "records_count": len(parsed_output.records_created),
        "record_ids": [r.record_id for r in parsed_output.records_created],
        "embedding_model": parsed_output.embedding_model,
        "status": parsed_output.status,
        "path": str(summary_path),
    }

    if parsed_output.status == "archived":
        event_name = "archivist_a1_archived"
        event_detail = (
            f"Archivist A1 created {len(parsed_output.records_created)} "
            f"record(s) at attempt {archival_attempt}"
        )
    else:
        event_name = "archivist_a1_error"
        event_detail = f"Archivist A1 archival error at attempt {archival_attempt}"

    return {
        "archived_summary_ref": new_archived_summary_ref,
        "status": parsed_output.status,
        "notes": parsed_output.notes,
        "audit": parsed_output.audit.model_dump() if parsed_output.audit else {},
        "increment_archival_attempt": True,
        "error_logs": error_logs,
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
