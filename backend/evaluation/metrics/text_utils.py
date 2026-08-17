"""
Small text-similarity helpers shared by every heuristic metric scorer
in `backend/evaluation/metrics/`. These are deliberately simple
(token overlap, not embeddings) so metrics have a zero-dependency
fallback whenever no LLM judge is available — see each metric module
for how this is combined with the LLM-judge path.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "but", "not",
    "this", "that", "these", "those", "it", "its", "as", "by", "from",
    "what", "which", "who", "whom", "does", "do", "did", "has", "have", "had",
}


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def token_overlap_f1(text_a: str, text_b: str) -> float:
    """Symmetric token-overlap score in [0, 1] (harmonic-mean style, like F1).

    Used as the offline stand-in for semantic similarity: not as good
    as an embedding or LLM judge, but dependency-free and directionally
    correct for meaningfully different vs. meaningfully similar text.
    """
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0

    overlap = len(tokens_a & tokens_b)
    precision = overlap / len(tokens_b)
    recall = overlap / len(tokens_a)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def coverage_ratio(claim_text: str, source_text: str) -> float:
    """Fraction of `claim_text`'s (non-stopword) tokens that also appear in `source_text`.

    Directional (not symmetric) — used for faithfulness/hallucination,
    where what matters is how much of the *answer* is grounded in the
    *context*, not the reverse.
    """
    claim_tokens = set(tokenize(claim_text))
    source_tokens = set(tokenize(source_text))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & source_tokens) / len(claim_tokens)
