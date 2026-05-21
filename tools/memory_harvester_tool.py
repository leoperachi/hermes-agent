#!/usr/bin/env python3
"""
Memory Harvester Tool

Scans Claude Code conversation files (~/.claude/projects/), extracts
learnings using an LLM, and persists them to the `learnings` table in
state.db.  Exposes a single `memory_harvest` tool the agent can call,
plus a `memory_search_learnings` tool for querying what was learned.

The harvester is idempotent: each conversation file is identified by its
SHA-256 hash and skipped if already recorded in `learnings_sources`.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core harvest logic (also callable programmatically)
# ---------------------------------------------------------------------------

def run_harvest(
    *,
    db=None,
    projects_root=None,
    max_conversations: int = 20,
    min_confidence: float = 0.5,
    min_human_turns: int = 2,
) -> Dict[str, Any]:
    """Discover new Claude Code conversations and extract learnings into state.db.

    Args:
        db:                   SessionDB instance. If None, opens the default DB.
        projects_root:        Override for the Claude Code projects directory.
        max_conversations:    Maximum number of conversations to process per run.
        min_confidence:       Minimum LLM confidence to persist a learning.
        min_human_turns:      Skip conversations with fewer human turns (too short).

    Returns:
        Summary dict with keys: conversations_scanned, conversations_processed,
        learnings_added, skipped, errors.
    """
    from tools.claude_conversation_reader import discover_conversations
    from tools.learning_extractor import extract_learnings

    if db is None:
        from hermes_state import SessionDB
        db = SessionDB()

    already_processed = db.get_processed_source_ids()

    conversations = discover_conversations(
        projects_root=projects_root,
        already_processed=already_processed,
        min_human_turns=min_human_turns,
    )
    conversations = conversations[:max_conversations]

    summary = {
        "conversations_scanned": len(conversations),
        "conversations_processed": 0,
        "learnings_added": 0,
        "skipped": 0,
        "errors": [],
    }

    for conv in conversations:
        try:
            transcript = conv.as_text
            if not transcript.strip():
                summary["skipped"] += 1
                db.mark_source_processed(
                    source_id=conv.file_hash,
                    source_type="claude_code",
                    file_path=conv.file_path,
                    learning_count=0,
                )
                continue

            learnings = extract_learnings(
                transcript=transcript,
                project_path=conv.project_path,
                min_confidence=min_confidence,
            )

            count = 0
            for learning in learnings:
                try:
                    db.add_learning(
                        source="claude_code",
                        source_id=conv.file_hash,
                        category=learning.category,
                        content=learning.content,
                        confidence=learning.confidence,
                        project_path=conv.project_path,
                    )
                    count += 1
                except Exception as exc:
                    logger.warning("Failed to persist learning: %s", exc)

            db.mark_source_processed(
                source_id=conv.file_hash,
                source_type="claude_code",
                file_path=conv.file_path,
                learning_count=count,
            )

            summary["conversations_processed"] += 1
            summary["learnings_added"] += count
            logger.info(
                "Processed %s → %d learnings", conv.session_id[:12], count
            )

        except Exception as exc:
            logger.error("Harvest failed for %s: %s", conv.file_path, exc)
            summary["errors"].append(str(exc))
            # Still mark as processed to avoid retry loops on broken files
            try:
                db.mark_source_processed(
                    source_id=conv.file_hash,
                    source_type="claude_code",
                    file_path=conv.file_path,
                    learning_count=0,
                )
            except Exception:
                pass

    return summary


def _handle_harvest(args: Dict[str, Any], **kwargs) -> str:
    """Tool handler for memory_harvest."""
    max_conv = min(int(args.get("max_conversations", 10)), 50)
    min_conf = max(0.0, min(1.0, float(args.get("min_confidence", 0.5))))

    db = kwargs.get("db") or kwargs.get("session_db")

    try:
        summary = run_harvest(
            db=db,
            max_conversations=max_conv,
            min_confidence=min_conf,
        )
        return json.dumps({
            "success": True,
            **summary,
            "message": (
                f"Processed {summary['conversations_processed']} conversation(s), "
                f"added {summary['learnings_added']} learning(s)."
            ),
        })
    except Exception as exc:
        logger.error("memory_harvest tool error: %s", exc, exc_info=True)
        return json.dumps({"success": False, "error": str(exc)})


def _handle_search_learnings(args: Dict[str, Any], **kwargs) -> str:
    """Tool handler for memory_search_learnings."""
    query = args.get("query", "").strip()
    limit = min(int(args.get("limit", 10)), 50)
    category = args.get("category", "").strip()

    db = kwargs.get("db") or kwargs.get("session_db")
    if db is None:
        try:
            from hermes_state import SessionDB
            db = SessionDB()
        except Exception as exc:
            return json.dumps({"success": False, "error": f"DB unavailable: {exc}"})

    try:
        if query:
            results = db.search_learnings(query, limit=limit)
        else:
            results = db.get_recent_learnings(limit=limit, category=category)

        # Bump used_count on returned results
        for r in results:
            try:
                db.bump_learning_used(r["id"])
            except Exception:
                pass

        return json.dumps({
            "success": True,
            "count": len(results),
            "learnings": [
                {
                    "id": r["id"],
                    "category": r["category"],
                    "content": r["content"],
                    "confidence": r["confidence"],
                    "project": r.get("project_path", ""),
                    "source": r["source"],
                    "created_at": r["created_at"],
                }
                for r in results
            ],
        })
    except Exception as exc:
        logger.error("memory_search_learnings tool error: %s", exc, exc_info=True)
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

MEMORY_HARVEST_SCHEMA = {
    "name": "memory_harvest",
    "description": (
        "Scan recent Claude Code conversations and extract learnings into the "
        "persistent memory database. "
        "Learnings capture technical decisions, user preferences, project context, "
        "and feedback from past sessions. "
        "Idempotent — already-processed conversations are skipped automatically.\n\n"
        "Run this periodically (e.g. at session start) or when the user asks you "
        "to 'learn from past conversations'.\n\n"
        "After harvesting, use memory_search_learnings to retrieve relevant learnings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "max_conversations": {
                "type": "integer",
                "description": "Maximum number of new conversations to process (default 10, max 50).",
                "default": 10,
            },
            "min_confidence": {
                "type": "number",
                "description": "Minimum LLM confidence (0.0-1.0) to persist a learning (default 0.5).",
                "default": 0.5,
            },
        },
        "required": [],
    },
}

MEMORY_SEARCH_LEARNINGS_SCHEMA = {
    "name": "memory_search_learnings",
    "description": (
        "Search the persistent learnings database for knowledge extracted from "
        "past Claude Code conversations.\n\n"
        "Use this to recall what was learned in previous sessions before starting "
        "a new task, or when the user references something discussed before.\n\n"
        "Three calling modes:\n"
        "  1) Search by query:    memory_search_learnings(query=\"docker networking\")\n"
        "  2) Browse by category: memory_search_learnings(category=\"user_pref\")\n"
        "  3) Browse recent:      memory_search_learnings(limit=20)\n\n"
        "Categories: technical, user_pref, project, feedback, general"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Full-text search query over learning content.",
            },
            "category": {
                "type": "string",
                "enum": ["technical", "user_pref", "project", "feedback", "general"],
                "description": "Filter by learning category.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return (default 10, max 50).",
                "default": 10,
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="memory_harvest",
    toolset="memory",
    schema=MEMORY_HARVEST_SCHEMA,
    handler=_handle_harvest,
)

registry.register(
    name="memory_search_learnings",
    toolset="memory",
    schema=MEMORY_SEARCH_LEARNINGS_SCHEMA,
    handler=_handle_search_learnings,
)
