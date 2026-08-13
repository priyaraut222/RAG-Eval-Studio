"""
Runs retrieval evaluation for a whole dataset against a given
retriever, producing per-item metrics plus dataset-level aggregates.

A dataset item's "relevant" chunk isn't an id — it's the free-text
`expected_chunk`/`expected_context` written or generated in the
Dataset Builder. So before we can compute Precision@K etc. we need to
decide which *retrieved* chunks count as matches for that expected
text. `is_relevant_chunk` does that with a token-overlap heuristic
rather than requiring exact string equality, since a chunker will
rarely reproduce the expected text byte-for-byte.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from backend.dataset.chunking import Chunk, chunk_text
from backend.dataset.schema import EvalDataset, EvalItem
from backend.retrieval.metrics import evaluate_ranking, mean_reciprocal_rank
from backend.retrieval.retriever import BaseRetriever, RetrievedChunk
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
RELEVANCE_OVERLAP_THRESHOLD = 0.5


def _tokens(text: str) -> set[str]:
    return set(t.lower() for t in _TOKEN_RE.findall(text))


def is_relevant_chunk(chunk_text_value: str, expected_chunk: str, expected_context: str) -> bool:
    """Heuristic match between a retrieved chunk and an item's expected text.

    A chunk counts as relevant if it shares at least
    `RELEVANCE_OVERLAP_THRESHOLD` of the expected text's tokens (or
    vice versa) — this tolerates different chunk boundaries while
    still requiring substantial topical overlap, rather than exact
    substring matching which would be too brittle.
    """
    expected = expected_chunk.strip() or expected_context.strip()
    if not expected or not chunk_text_value.strip():
        return False

    expected_tokens = _tokens(expected)
    chunk_tokens = _tokens(chunk_text_value)
    if not expected_tokens or not chunk_tokens:
        return False

    overlap = len(expected_tokens & chunk_tokens)
    smaller = min(len(expected_tokens), len(chunk_tokens))
    return (overlap / smaller) >= RELEVANCE_OVERLAP_THRESHOLD


@dataclass
class ItemRetrievalResult:
    """Per-item retrieval outcome: what was retrieved, and how it scored."""

    item_id: str
    question: str
    retrieved: list[RetrievedChunk]
    relevant_chunk_ids: set[str]
    metrics: dict[str, float]
    latency_ms: float


@dataclass
class RetrievalEvalReport:
    """Full dataset-level retrieval evaluation result."""

    retriever_name: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    per_item: list[ItemRetrievalResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)

    @property
    def worst_items(self) -> list[ItemRetrievalResult]:
        return sorted(self.per_item, key=lambda r: r.metrics.get("ndcg_at_k", 0.0))

    @property
    def best_items(self) -> list[ItemRetrievalResult]:
        return sorted(self.per_item, key=lambda r: r.metrics.get("ndcg_at_k", 0.0), reverse=True)


def evaluate_dataset_retrieval(
    dataset: EvalDataset,
    retriever: BaseRetriever,
    source_text: str,
    top_k: int = 5,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> RetrievalEvalReport:
    """Chunk `source_text`, index it in `retriever`, then evaluate every dataset item's retrieval.

    `source_text` is the corpus the retriever searches over — typically
    the same document(s) the dataset's questions were built from.
    """
    chunks: list[Chunk] = chunk_text(source_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No text to index — source_text is empty after chunking.")

    retriever.index(chunks)

    per_item: list[ItemRetrievalResult] = []
    per_query_retrieved: list[list[str]] = []
    per_query_relevant: list[set[str]] = []

    for item in dataset.items:
        if not item.is_complete():
            continue

        start = time.perf_counter()
        retrieved = retriever.retrieve(item.question, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        relevant_ids = {
            hit.chunk_id
            for hit in retrieved
            if is_relevant_chunk(hit.text, item.expected_chunk, item.expected_context)
        }
        # Also check chunks that scored outside the retrieved set, so recall
        # reflects chunks that exist in the corpus but weren't retrieved —
        # otherwise recall would trivially always be 1.0 or based only on
        # retrieved items.
        all_relevant_ids = {
            f"chunk_{c.chunk_index}"
            for c in chunks
            if is_relevant_chunk(c.text, item.expected_chunk, item.expected_context)
        }
        relevant_ids = relevant_ids | all_relevant_ids

        retrieved_ids = [hit.chunk_id for hit in retrieved]
        metrics = evaluate_ranking(retrieved_ids, relevant_ids, k=top_k)

        per_item.append(
            ItemRetrievalResult(
                item_id=item.id,
                question=item.question,
                retrieved=retrieved,
                relevant_chunk_ids=relevant_ids,
                metrics=metrics,
                latency_ms=latency_ms,
            )
        )
        per_query_retrieved.append(retrieved_ids)
        per_query_relevant.append(relevant_ids)

    aggregate = _aggregate_metrics(per_item)
    aggregate["mrr"] = mean_reciprocal_rank(per_query_retrieved, per_query_relevant)

    return RetrievalEvalReport(
        retriever_name=retriever.name,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        per_item=per_item,
        aggregate=aggregate,
    )


def _aggregate_metrics(per_item: list[ItemRetrievalResult]) -> dict[str, float]:
    if not per_item:
        return {}
    keys = per_item[0].metrics.keys()
    return {key: sum(r.metrics[key] for r in per_item) / len(per_item) for key in keys}
