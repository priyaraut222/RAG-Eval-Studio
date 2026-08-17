"""
Shared "ask an LLM to score this and explain why" helper used by every
metric that has an LLM-judge path. Centralizing this means each metric
module only needs to supply its own prompt text — parsing, error
handling, and the local-provider bypass live in one place.
"""

from __future__ import annotations

import json
import re

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator of RAG (Retrieval-Augmented Generation) system "
    "outputs. You always respond with a single JSON object and nothing else: "
    '{"score": <float between 0.0 and 1.0>, "reasoning": "<one or two sentence explanation>"}. '
    "Do not include markdown code fences or any text outside the JSON object."
)


def parse_judge_response(text: str) -> tuple[float, str]:
    """Extract (score, reasoning) from a judge LLM's raw text output.

    Tolerates markdown code fences and stray text around the JSON
    object; raises `ValueError` if no usable score can be found so
    callers can fall back to a heuristic instead of trusting garbage.
    """
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:200]!r}")

    data = json.loads(match.group(0))
    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))  # clamp defensively
    reasoning = str(data.get("reasoning", "")).strip()
    return score, reasoning


def call_judge(client, user_prompt: str) -> tuple[float, str] | None:
    """Run the LLM-judge path, or return None to signal "use the heuristic instead".

    Returns None when the client is on the local/offline provider
    (no point prompting a rule-based stand-in for a subjective 0-1
    judgment) or when the call/parse fails for any reason — the
    caller's heuristic fallback handles both cases identically.
    """
    if getattr(client, "active_provider", "local") == "local":
        return None

    try:
        response = client.complete(prompt=user_prompt, system=JUDGE_SYSTEM_PROMPT, temperature=0.0)
        return parse_judge_response(response.text)
    except Exception as exc:
        logger.warning(f"LLM judge call failed, falling back to heuristic: {exc}")
        return None
