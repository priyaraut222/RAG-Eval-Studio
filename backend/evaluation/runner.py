"""
Orchestrates Phase 4: for every item in a dataset, generate an answer
from its expected context, score it on all six LLM evaluation
metrics, and track latency/tokens/cost — then aggregate across the
whole dataset. This is what `app/pages/llm_evaluation.py` calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from backend.config.pricing import estimate_cost
from backend.dataset.schema import EvalDataset
from backend.evaluation.metrics.base import MetricResult
from backend.evaluation.metrics.registry import get_all_metrics
from backend.evaluation.pipeline import generate_answer
from backend.utils.llm_client import LLMClient
from backend.utils.logger import get_logger

logger = get_logger(__name__)

ContextSource = Literal["expected_context", "expected_chunk"]


@dataclass
class ItemLLMResult:
    """Per-item outcome: the generated answer, every metric's score, and cost/latency."""

    item_id: str
    question: str
    ground_truth: str
    context: str
    generated_answer: str
    metrics: dict[str, MetricResult]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost: float

    def overall_score(self) -> float:
        """Mean of the "higher is better" metrics — Hallucination is excluded (it's inverted)."""
        scoring_metrics = [m for m in self.metrics.values() if m.key != "hallucination"]
        if not scoring_metrics:
            return 0.0
        return sum(m.score for m in scoring_metrics) / len(scoring_metrics)


@dataclass
class LLMEvalReport:
    """Full dataset-level LLM evaluation result."""

    provider: str
    model: str
    context_source: str
    per_item: list = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def worst_items(self):
        return sorted(self.per_item, key=lambda r: r.overall_score())

    @property
    def best_items(self):
        return sorted(self.per_item, key=lambda r: r.overall_score(), reverse=True)


def run_llm_evaluation(
    dataset: EvalDataset,
    client: LLMClient,
    context_source: ContextSource = "expected_context",
) -> LLMEvalReport:
    """Generate + score every complete item in `dataset`, returning a full report."""
    metrics = get_all_metrics()
    per_item: list[ItemLLMResult] = []

    for item in dataset.items:
        if not item.is_complete():
            continue

        context_text = getattr(item, context_source, "") or item.expected_context or item.expected_chunk
        if not context_text:
            continue

        start = time.perf_counter()
        answer, response = generate_answer(item.question, context_text, client)
        latency_ms = (time.perf_counter() - start) * 1000

        item_metrics: dict[str, MetricResult] = {}
        for metric in metrics:
            try:
                item_metrics[metric.key] = metric.compute(
                    question=item.question,
                    answer=answer,
                    contexts=[context_text],
                    ground_truth=item.ground_truth,
                    client=client,
                )
            except Exception as exc:
                logger.warning(f"Metric '{metric.key}' failed for item {item.id}: {exc}")
                item_metrics[metric.key] = MetricResult(metric.key, metric.label, 0.0, f"Scoring failed: {exc}", method="error")

        input_tokens = response.input_tokens if response else max(len(context_text) // 4, 1)
        output_tokens = response.output_tokens if response else max(len(answer) // 4, 1)
        model_name = response.model if response else "heuristic-v1"
        cost = estimate_cost(model_name, input_tokens, output_tokens)

        per_item.append(
            ItemLLMResult(
                item_id=item.id,
                question=item.question,
                ground_truth=item.ground_truth,
                context=context_text,
                generated_answer=answer,
                metrics=item_metrics,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=cost,
            )
        )

    aggregate = _aggregate_metrics(per_item, metrics)
    total_cost = sum(r.estimated_cost for r in per_item)
    avg_latency = sum(r.latency_ms for r in per_item) / len(per_item) if per_item else 0.0

    # Report the model actually used for generation. `LLMClient` is a facade and
    # doesn't expose the model name directly, so we resolve it via a small helper.
    model_name = _resolve_model_name(client)

    return LLMEvalReport(
        provider=client.active_provider,
        model=model_name,
        context_source=context_source,
        per_item=per_item,
        aggregate=aggregate,
        total_cost=total_cost,
        avg_latency_ms=avg_latency,
        total_input_tokens=sum(r.input_tokens for r in per_item),
        total_output_tokens=sum(r.output_tokens for r in per_item),
    )


def _resolve_model_name(client: LLMClient) -> str:
    """Best-effort model name for the report header.

    `LLMClient` is a provider-agnostic facade and doesn't expose the
    underlying model name itself, so we ask it directly via its
    `model_name` property (added alongside `active_provider`).
    """
    return client.model_name


def _aggregate_metrics(per_item: list, metrics) -> dict:
    if not per_item:
        return {}
    aggregate = {}
    for metric in metrics:
        scores = [r.metrics[metric.key].score for r in per_item if metric.key in r.metrics]
        aggregate[metric.key] = sum(scores) / len(scores) if scores else 0.0
    return aggregate
