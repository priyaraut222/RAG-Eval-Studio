"""
Retrieval quality metrics.

Every function here takes a ranked list of retrieved chunk/doc ids and
the set of ids that are actually relevant to the query, and returns a
score in [0, 1] (nDCG when using graded relevance can exceed slightly
due to floating point, but is clamped). These are pure, framework-free
functions so they're trivially unit-testable and reusable from both
the Retrieval Evaluation page and the Experiment runner.

All functions are null-safe: an empty `relevant_ids` set (a dataset
item with no known-relevant chunk) returns 0.0 rather than raising,
since "nothing was relevant so nothing could be found" is a real,
scoreable outcome, not an error state.
"""

from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def hit_rate(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if at least one relevant item is in the top-k, else 0.0."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / (rank of the first relevant item), or 0.0 if none is retrieved.

    Ranks are 1-indexed, matching the standard MRR definition.
    """
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(per_query_retrieved: list[list[str]], per_query_relevant: list[set[str]]) -> float:
    """MRR across a batch of queries — the mean of `reciprocal_rank` per query."""
    if not per_query_retrieved:
        return 0.0
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(per_query_retrieved, per_query_relevant)
    ]
    return sum(scores) / len(scores)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int, graded_relevance: dict[str, float] | None = None) -> float:
    """Normalized Discounted Cumulative Gain at k.

    With no `graded_relevance` map, relevance is binary (1.0 if the id
    is in `relevant_ids`, else 0.0) — the common case when a dataset
    item has one expected chunk rather than graded relevance labels.
    """
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    def _relevance(doc_id: str) -> float:
        if graded_relevance is not None:
            return graded_relevance.get(doc_id, 0.0)
        return 1.0 if doc_id in relevant_ids else 0.0

    dcg = sum(_relevance(doc_id) / math.log2(rank + 1) for rank, doc_id in enumerate(top_k, start=1))

    # Ideal ranking: relevant items sorted by relevance, best first.
    if graded_relevance is not None:
        ideal_relevances = sorted((graded_relevance.get(d, 0.0) for d in relevant_ids), reverse=True)
    else:
        ideal_relevances = [1.0] * len(relevant_ids)
    ideal_relevances = ideal_relevances[:k]
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))

    if idcg == 0.0:
        return 0.0
    return min(dcg / idcg, 1.0)


def evaluate_ranking(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> dict[str, float]:
    """Compute the full retrieval-metric suite for a single query's ranking."""
    return {
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "hit_rate": hit_rate(retrieved_ids, relevant_ids, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, relevant_ids, k),
    }
