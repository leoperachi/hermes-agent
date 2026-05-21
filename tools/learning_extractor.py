#!/usr/bin/env python3
"""
Learning Extractor

Uses an LLM to extract structured learnings from conversation transcripts.
Produces a list of Learning objects with category, content, and confidence.

Categories:
  - technical:    code patterns, architecture decisions, tool quirks, bugs fixed
  - user_pref:    user preferences, communication style, workflow habits
  - project:      project context, goals, constraints, decisions
  - feedback:     what worked / what didn't, corrections given

The extractor resolves provider credentials from the Hermes config, so it
runs with whichever model the user has set up. Falls back to OPENAI_API_KEY
if Hermes config is unavailable (e.g. standalone use).
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"technical", "user_pref", "project", "feedback", "general"}

_EXTRACTION_SYSTEM = """\
You are a knowledge extraction assistant. Extract learnings from a conversation transcript.

IMPORTANT: Respond with ONLY a valid JSON array. No prose, no markdown, no explanations before or after.

Each item in the array must have exactly these fields:
  "category": one of "technical", "user_pref", "project", "feedback", "general"
  "content": a self-contained statement, max 200 chars
  "confidence": a float 0.0-1.0

A learning is a specific, actionable insight useful in future conversations — not vague observations.

Categories:
- technical:  code patterns, tool quirks, architecture decisions, bugs/fixes
- user_pref:  preferences, workflow habits, liked/disliked approaches
- project:    project goals, constraints, current focus
- feedback:   corrections ("don't do X") or confirmed approaches ("yes, exactly")
- general:    any other useful insight

The transcript may contain [ASSISTANT REASONING] blocks showing the model's chain-of-thought
before giving a response. These blocks contain valuable context about WHY decisions were made —
extract learnings from them too, especially architectural reasoning and preference signals.

Output format (ONLY this, nothing else):
[
  {"category": "technical", "content": "...", "confidence": 0.9},
  {"category": "user_pref", "content": "...", "confidence": 0.8}
]

If there are no meaningful learnings, output exactly: []\
"""

_EXTRACTION_USER_TMPL = """\
Project: {project_path}

Conversation transcript:
---
{transcript}
---

Extract learnings from this conversation.\
"""


@dataclass
class Learning:
    category: str
    content: str
    confidence: float = 0.8
    extractor_model: str = ""

    def is_valid(self) -> bool:
        return (
            self.category in VALID_CATEGORIES
            and bool(self.content.strip())
            and 0.0 <= self.confidence <= 1.0
            and len(self.content) <= 500
        )


def _resolve_llm_credentials() -> Tuple[str, str, str]:
    """Return (api_key, base_url, model) for a lightweight extraction call.

    Tries in order:
    1. Hermes config — uses the configured primary provider
    2. Ollama local (http://localhost:11434/v1 or OLLAMA_HOST env var)
    3. Environment variables (OPENAI_API_KEY / OPENAI_BASE_URL)
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        model_cfg = cfg.get("model", {})
        model = ""
        if isinstance(model_cfg, dict):
            model = model_cfg.get("default") or model_cfg.get("model", "")
        elif isinstance(model_cfg, str):
            model = model_cfg

        from hermes_cli.auth import resolve_api_key_provider_credentials
        provider_id = cfg.get("provider", "")
        if provider_id:
            creds = resolve_api_key_provider_credentials(provider_id)
            api_key = creds.get("api_key", "")
            base_url = creds.get("base_url", "")
            if api_key and model:
                return api_key, base_url, model
    except Exception as exc:
        logger.debug("Could not read Hermes config for LLM creds: %s", exc)

    # Ollama: check common hosts (env var > localhost > WSL host)
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    ollama_candidates = []
    if ollama_host:
        ollama_candidates.append(ollama_host.rstrip("/") + "/v1")
    ollama_candidates += [
        "http://localhost:11434/v1",
        "http://172.22.240.1:11434/v1",
    ]
    for candidate in ollama_candidates:
        try:
            import urllib.request
            tags_url = candidate.replace("/v1", "/api/tags")
            urllib.request.urlopen(tags_url, timeout=2)
            logger.debug("Using Ollama at %s model=%s", candidate, ollama_model)
            return "ollama", candidate, ollama_model
        except Exception:
            continue

    # Fallback: standard OpenAI env vars
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("EXTRACTION_MODEL", "gpt-4o-mini")
    return api_key, base_url, model


def _call_llm(prompt: str, api_key: str, base_url: str, model: str,
              max_retries: int = 3) -> str:
    """Call the LLM and return the raw response text.

    Retries up to max_retries times on transient errors (rate limits,
    connection errors, timeouts) with exponential backoff + jitter.
    Non-transient errors (auth, bad model name) are raised immediately.
    """
    from openai import OpenAI, APIConnectionError, RateLimitError, APITimeoutError
    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=60,
        max_retries=1,
    )
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            return response.choices[0].message.content or ""
        except (APIConnectionError, RateLimitError, APITimeoutError) as exc:
            last_exc = exc
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Learning extractor: transient LLM error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
        except Exception:
            # Non-transient (auth failure, invalid model, etc.) — raise immediately
            raise
    raise last_exc


def _parse_learnings(raw: str) -> List[Learning]:
    """Parse LLM JSON output into Learning objects, tolerating minor deviations."""
    raw = raw.strip()

    # Strip <think>...</think> blocks (chain-of-thought models like qwen3)
    import re as _re
    raw = _re.sub(r'<think>[\s\S]*?</think>', '', raw, flags=_re.IGNORECASE).strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    # If still not parseable, try to extract a JSON array from anywhere in the text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        import re as _re2
        match = _re2.search(r'\[[\s\S]*\]', raw)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                logger.warning("Learning extractor: JSON parse error: %s — raw: %.200s", exc, raw)
                return []
        else:
            logger.warning("Learning extractor: no JSON array found — raw: %.200s", raw)
            return []

    if not isinstance(data, list):
        logger.warning("Learning extractor: expected JSON array, got %s", type(data))
        return []

    learnings = []
    for item in data:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "general")).lower().strip()
        if category not in VALID_CATEGORIES:
            category = "general"
        content = str(item.get("content", "")).strip()
        try:
            confidence = float(item.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.8

        learning = Learning(category=category, content=content, confidence=confidence)
        if learning.is_valid():
            learnings.append(learning)

    return learnings


def extract_learnings(
    transcript: str,
    project_path: str = "",
    min_confidence: float = 0.5,
    credentials: Optional[Tuple[str, str, str]] = None,
) -> List[Learning]:
    """Extract learnings from a conversation transcript.

    Args:
        transcript:      Plain-text conversation (role-prefixed turns).
        project_path:    Project directory context for the LLM prompt.
        min_confidence:  Minimum confidence threshold to keep a learning.
        credentials:     Optional override (api_key, base_url, model).

    Returns:
        List of Learning objects above the confidence threshold.
    """
    if not transcript or len(transcript.strip()) < 100:
        return []

    # Truncate very long transcripts to avoid token limits
    max_chars = 12_000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n[transcript truncated]"

    if credentials is None:
        api_key, base_url, model = _resolve_llm_credentials()
    else:
        api_key, base_url, model = credentials

    if not api_key:
        logger.warning("Learning extractor: no API key available — skipping extraction")
        return []

    prompt = _EXTRACTION_USER_TMPL.format(
        project_path=project_path or "(unknown)",
        transcript=transcript,
    )

    try:
        raw = _call_llm(prompt, api_key, base_url, model)
    except Exception as exc:
        logger.warning("Learning extractor: LLM call failed: %s", exc)
        return []

    learnings = _parse_learnings(raw)
    for learning in learnings:
        learning.extractor_model = model
    filtered = [l for l in learnings if l.confidence >= min_confidence]
    logger.info(
        "Extracted %d learnings (from %d) for project=%s model=%s",
        len(filtered), len(learnings), project_path or "(unknown)", model,
    )
    return filtered
