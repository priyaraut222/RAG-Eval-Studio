"""
A lightweight, dependency-free reranker.

Real RAG pipelines often rerank a retriever's top candidates with a
cross-encoder for better precision. Running an actual cross-encoder
model is out of scope for an offline-friendly demo, so this reranker
uses a different, complementary signal — direct query/chunk token
overlap — to reorder candidates. It's not a cross-encoder, but it IS
a genuine second scoring pass that can and does change the final
ranking, which is what the "with/without reranker" comparison in the
Experiment Manager needs to be honest.
"""

from __future__ import annotations

from backend.evaluation.metrics.text_utils import token_overlap_f1
from backend.retrieval.retriever import RetrievedChunk


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score `candidates` by query/chunk token overlap and return the new top-k.

    Callers should over-fetch from the retriever (e.g. `top_k * 3`)
    before calling this, so there's a real candidate pool to rerank
    rather than just re-sorting an already-truncated top-k.
    """
    if not candidates:
        return []

    rescored = [
        RetrievedChunk(chunk_id=c.chunk_id, text=c.text, score=token_overlap_f1(query, c.text))
        for c in candidates
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]
