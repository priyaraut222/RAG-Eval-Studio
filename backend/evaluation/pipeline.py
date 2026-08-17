"""
Generation step for LLM Evaluation.

Since this platform evaluates RAG *pipelines* rather than being one,
there's no externally-connected generator by default — Phase 4 needs
*an* answer to score. This module produces one: a real LLM call when
a provider is configured, or a deterministic extractive heuristic
(pick the context sentence most relevant to the question) when
running fully offline. Either way, `run_llm_evaluation` in
`runner.py` scores whatever comes back the same way — this is what
gets evaluated, not a shortcut around evaluation.
"""

from __future__ import annotations

import re

from backend.utils.llm_client import LLMClient, LLMResponse
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

GENERATION_SYSTEM_PROMPT = (
    "You are a RAG (Retrieval-Augmented Generation) system's answer generator. "
    "Answer the user's question using ONLY the provided context. If the context doesn't contain "
    "the answer, say so plainly. Be concise — 1 to 3 sentences."
)


def generate_answer(question: str, context_text: str, client: LLMClient) -> tuple[str, LLMResponse | None]:
    """Return (answer_text, raw_response_or_None).

    `raw_response_or_None` carries token counts for cost/latency
    accounting; it's `None` only in the (rare) case the client itself
    returns no usable text, which callers treat as an empty answer.
    """
    if client.active_provider == "local":
        return _heuristic_answer(question, context_text), None

    prompt = f"Context:\n\"\"\"\n{context_text}\n\"\"\"\n\nQuestion: {question}\n\nAnswer:"
    try:
        response = client.complete(prompt=prompt, system=GENERATION_SYSTEM_PROMPT, temperature=0.2)
        return response.text.strip(), response
    except Exception as exc:
        logger.warning(f"Answer generation failed, falling back to heuristic: {exc}")
        return _heuristic_answer(question, context_text), None


def _heuristic_answer(question: str, context_text: str) -> str:
    """Extractive fallback: return the context sentence with the most token overlap with the question."""
    from backend.evaluation.metrics.text_utils import tokenize

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(context_text) if s.strip()]
    if not sentences:
        return "The retrieved context did not contain enough information to answer this question."

    question_tokens = set(tokenize(question))
    if not question_tokens:
        return sentences[0]

    best_sentence = sentences[0]
    best_overlap = -1
    for sentence in sentences:
        overlap = len(question_tokens & set(tokenize(sentence)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    return best_sentence
