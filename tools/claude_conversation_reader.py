#!/usr/bin/env python3
"""
Claude Code Conversation Reader

Discovers and parses JSONL conversation files written by Claude Code CLI
(~/.claude/projects/<encoded-path>/*.jsonl) and normalizes them into a
format suitable for learning extraction.

Each JSONL file contains one message per line. Relevant line types:
  - type == "user":      human turn
  - type == "assistant": model turn (may contain text or tool_use blocks)

Queue-operation and summary lines are skipped.

The reader tracks which files have already been processed via the
`learnings_sources` table in state.db (populated by memory_harvester_tool).
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    role: str          # "user" or "assistant"
    text: str          # plain text content (tool calls stripped)
    timestamp: str     # ISO8601
    uuid: str
    reasoning: str = ""  # content of thinking/reasoning blocks


@dataclass
class Conversation:
    file_path: str
    file_hash: str      # sha256 of file contents — used as source_id
    session_id: str     # from filename stem
    project_path: str   # decoded from parent directory name
    messages: List[ConversationMessage] = field(default_factory=list)

    @property
    def human_turns(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    @property
    def as_text(self) -> str:
        """Plain-text transcript, suitable for LLM processing."""
        parts = []
        for msg in self.messages:
            if msg.reasoning.strip():
                parts.append(f"[{msg.role.upper()} REASONING]\n{msg.reasoning.strip()}")
            if msg.text.strip():
                parts.append(f"[{msg.role.upper()}]\n{msg.text.strip()}")
        return "\n\n".join(parts)


def _decode_project_path(dir_name: str) -> str:
    """Reverse Claude Code's path encoding: '--' prefix + '/' → '-'."""
    # Claude Code encodes absolute paths by replacing / with - and prepending --
    # e.g. /home/user/myproject → --home-user-myproject
    if dir_name.startswith("--"):
        # Replace - with / but keep -- prefix stripped
        return "/" + dir_name[2:].replace("-", "/")
    return dir_name


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from a message content value.

    Content can be:
      - str (legacy / simple assistant messages)
      - list of blocks: {"type": "text", "text": "..."} or tool_use blocks
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        # tool_use and tool_result blocks are intentionally skipped
    return "\n".join(parts)


def _extract_reasoning_from_content(content: Any) -> str:
    """Extract reasoning/thinking block text from a message content value.

    Handles Claude API native thinking blocks: {"type": "thinking", "thinking": "..."}
    """
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                parts.append(thinking)
    return "\n".join(parts)


def _file_hash(path: Path) -> str:
    """SHA-256 of file contents, hex-encoded."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_jsonl(path: Path) -> List[ConversationMessage]:
    """Parse a Claude Code JSONL file into ConversationMessage list."""
    messages: List[ConversationMessage] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type", "")
                if entry_type not in ("user", "assistant"):
                    continue

                msg_obj = entry.get("message", {})
                role = msg_obj.get("role", entry_type)
                content = msg_obj.get("content", "")
                text = _extract_text_from_content(content)
                reasoning = _extract_reasoning_from_content(content)
                if not text.strip() and not reasoning.strip():
                    continue

                messages.append(ConversationMessage(
                    role=role,
                    text=text,
                    timestamp=entry.get("timestamp", ""),
                    uuid=entry.get("uuid", ""),
                    reasoning=reasoning,
                ))
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
    return messages


def _find_claude_projects_root() -> Optional[Path]:
    """Locate ~/.claude/projects, checking both native home and common WSL mounts."""
    candidates = [
        Path.home() / ".claude" / "projects",
    ]
    # WSL: Windows home may differ from Linux home
    win_user = os.environ.get("USERPROFILE", "").replace("\\", "/")
    if win_user:
        candidates.append(Path(win_user.replace("C:/", "/mnt/c/")) / ".claude" / "projects")
    # Also check standard Windows mounts
    for drive in ("c", "d"):
        for username_env in ("USERNAME", "USER"):
            uname = os.environ.get(username_env, "")
            if uname:
                candidates.append(
                    Path(f"/mnt/{drive}/Users/{uname}/.claude/projects")
                )

    for candidate in candidates:
        if candidate.is_dir():
            logger.debug("Found Claude projects at %s", candidate)
            return candidate
    return None


def discover_conversations(
    projects_root: Optional[Path] = None,
    already_processed: Optional[set] = None,
    min_human_turns: int = 2,
) -> List[Conversation]:
    """Scan the Claude Code projects directory and return parsed conversations.

    Args:
        projects_root:      Override the auto-detected ~/.claude/projects path.
        already_processed:  Set of file_hash strings to skip (already in DB).
        min_human_turns:    Skip very short files with fewer human turns.

    Returns:
        List of Conversation objects, newest files first.
    """
    if already_processed is None:
        already_processed = set()

    root = projects_root or _find_claude_projects_root()
    if root is None:
        logger.warning("Claude Code projects directory not found")
        return []

    conversations: List[Conversation] = []
    for jsonl_path in sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            fhash = _file_hash(jsonl_path)
        except OSError:
            continue

        if fhash in already_processed:
            continue

        project_dir = jsonl_path.parent.name
        project_path = _decode_project_path(project_dir)
        session_id = jsonl_path.stem

        messages = _parse_jsonl(jsonl_path)
        conv = Conversation(
            file_path=str(jsonl_path),
            file_hash=fhash,
            session_id=session_id,
            project_path=project_path,
            messages=messages,
        )

        if conv.human_turns < min_human_turns:
            logger.debug("Skipping %s — only %d human turns", jsonl_path.name, conv.human_turns)
            continue

        conversations.append(conv)

    logger.info("Discovered %d new conversations to process", len(conversations))
    return conversations
