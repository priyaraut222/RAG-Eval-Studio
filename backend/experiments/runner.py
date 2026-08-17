"""
Runs one `ExperimentConfig` against a dataset, end to end: chunk the
source text, build the configured retriever (+ optional reranker),
retrieve real context per question, generate an answer from that
context, score it on every retrieval + LLM metric, and track
cost/latency — then aggregate into an `ExperimentRun`.

This is deliberately a *real* retrieval pass (unlike the Phase 4 LLM
Evaluation page, which can shortcut straight to `expected_context`)
so that chunk size, retriever choice, vector store, Top-K, and
reranker on/off all actually affect the result — that's the whole
point of comparing configurations.
"""

from __future__ import annotations

import time

from backend.config.pricing import estimate_cost
from backend.config.settings import get_settings
from backend.dataset.chunking import chunk_text
from backend.dataset.schema import EvalDataset
from backend.evaluation.metrics.registry import get_all_metrics
from backend.evaluation.pipeline import generate_answer
from backend.experiments.config import ExperimentConfig
from backend.experiments.results import ExperimentRun, ItemResult
from backend.retrieval.evaluator import is_relevant_chunk
from backend.retrieval.metrics import evaluate_ranking, mean_reciprocal_rank
from backend.retrieval.reranker import rerank
from backend.retrieval.retriever import get_retriever
from backend.utils.llm_client import get_llm_client
from backend.utils.logger import get_logger

logger = get_logger(__name__)

RERANK_CANDIDATE_MULTIPLIER = 3


def run_experiment(dataset: EvalDataset, config: ExperimentConfig, source_text: str) -> ExperimentRun:
    """Execute `config` against every complete item in `dataset`, returning a full report."""
    settings = get_settings()
    settings.default_llm_provider = config.llm_provider  # type: ignore[assignment]
    client = get_llm_client(settings)

    chunks = chunk_text(source_text, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    if not chunks:
        raise ValueError("No text to index — source_text is empty after chunking.")

    retriever = get_retriever(config.retriever, embedding_model=config.embedding_model, vector_store_name=config.vector_store)
    retriever.index(chunks)

    metrics = get_all_metrics()
    items: list[ItemResult] = []
    per_query_retrieved: list[list[str]] = []
    per_query_relevant: list[set[str]] = []

    for item in dataset.items:
        if not item.is_complete():
            continue

        # --- Retrieval (with optional rerank pass) ---
        fetch_k = config.top_k * RERANK_CANDIDATE_MULTIPLIER if config.use_reranker else config.top_k
        candidates = retriever.retrieve(item.question, top_k=fetch_k)
        retrieved = rerank(item.question, candidates, config.top_k) if config.use_reranker else candidates[: config.top_k]

        relevant_ids = {
            hit.chunk_id for hit in retrieved if is_relevant_chunk(hit.text, item.expected_chunk, item.expected_context)
        } | {
            f"chunk_{c.chunk_index}"
            for c in chunks
            if is_relevant_chunk(c.text, item.expected_chunk, item.expected_context)
        }
        retrieved_ids = [hit.chunk_id for hit in retrieved]
        retrieval_metrics = evaluate_ranking(retrieved_ids, relevant_ids, k=config.top_k)
        per_query_retrieved.append(retrieved_ids)
        per_query_relevant.append(relevant_ids)

        context_text = "\n\n".join(hit.text for hit in retrieved) or "(no context retrieved)"

        # --- Generation ---
        start = time.perf_counter()
        answer, response = generate_answer(item.question, context_text, client)
        latency_ms = (time.perf_counter() - start) * 1000

        # --- LLM metrics ---
        llm_scores: dict[str, float] = {}
        llm_reasoning: dict[str, str] = {}
        for metric in metrics:
            try:
                result = metric.compute(
                    question=item.question,
                    answer=answer,
                    contexts=[hit.text for hit in retrieved] or [context_text],
                    ground_truth=item.ground_truth,
                    client=client,
                )
                llm_scores[metric.key] = result.score
                llm_reasoning[metric.key] = result.reasoning
            except Exception as exc:
                logger.warning(f"Metric '{metric.key}' failed for item {item.id} under config '{config.name}': {exc}")
                llm_scores[metric.key] = 0.0
                llm_reasoning[metric.key] = f"Scoring failed: {exc}"

        input_tokens = response.input_tokens if response else max(len(context_text) // 4, 1)
        output_tokens = response.output_tokens if response else max(len(answer) // 4, 1)
        model_name = response.model if response else "heuristic-v1"
        cost = estimate_cost(model_name, input_tokens, output_tokens)

        items.append(
            ItemResult(
                item_id=item.id,
                question=item.question,
                ground_truth=item.ground_truth,
                retrieved_context=context_text,
                generated_answer=answer,
                retrieval_metrics=retrieval_metrics,
                llm_scores=llm_scores,
                llm_reasoning=llm_reasoning,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=cost,
            )
        )

    aggregate_retrieval = _aggregate(items, lambda r: r.retrieval_metrics)
    if per_query_retrieved:
        aggregate_retrieval["mrr"] = mean_reciprocal_rank(per_query_retrieved, per_query_relevant)
    aggregate_llm = _aggregate(items, lambda r: r.llm_scores)

    return ExperimentRun(
        config=config,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        items=items,
        aggregate_retrieval=aggregate_retrieval,
        aggregate_llm=aggregate_llm,
        total_cost=sum(r.estimated_cost for r in items),
        avg_latency_ms=sum(r.latency_ms for r in items) / len(items) if items else 0.0,
        total_input_tokens=sum(r.input_tokens for r in items),
        total_output_tokens=sum(r.output_tokens for r in items),
    )


def _aggregate(items: list[ItemResult], getter) -> dict[str, float]:
    if not items:
        return {}
    keys = getter(items[0]).keys()
    return {key: sum(getter(r).get(key, 0.0) for r in items) / len(items) for key in keys}
