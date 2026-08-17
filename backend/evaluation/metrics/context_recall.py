"""
Context Recall — whether the retrieved context contains everything
needed to reconstruct the ground-truth answer, i.e. did the retriever
miss information the generator would have needed.
"""

from __future__ import annotations

from backend.evaluation.metrics.base import BaseMetric, MetricExplainer, MetricResult
from backend.evaluation.metrics.judge import call_judge
from backend.evaluation.metrics.text_utils import coverage_ratio


class ContextRecallMetric(BaseMetric):
    key = "context_recall"
    label = "Context Recall"
    higher_is_better = True
    explainer = MetricExplainer(
        what="How much of the ground-truth answer's content is actually present in the retrieved context.",
        why="Even a highly faithful, well-phrased answer can't include information the retriever never "
        "surfaced. Low context recall points to a retrieval gap — the fix is chunking, embeddings, or Top-K, "
        "not the generation prompt.",
        how="An LLM judge compares the ground-truth answer against the retrieved context and scores what "
        "fraction of the ground truth's claims are supported by that context. Offline, a token-coverage "
        "heuristic checks what share of the ground truth's content words appear in the retrieved context.",
    )

    def compute(self, question: str, answer: str, contexts: list[str], ground_truth: str, client) -> MetricResult:
        context_text = "\n\n".join(contexts)
        if not context_text.strip():
            return MetricResult(self.key, self.label, 0.0, "No context was retrieved for this question.", method="heuristic")

        judged = call_judge(
            client,
            f"Ground truth answer:\n\"\"\"\n{ground_truth}\n\"\"\"\n\n"
            f"Retrieved context:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            "Score what fraction of the ground truth answer's content is supported by the retrieved context: "
            "1.0 means the context contains everything needed to produce the ground truth, 0.0 means none of it.",
        )
        if judged is not None:
            score, reasoning = judged
            return MetricResult(self.key, self.label, score, reasoning, method="llm_judge")

        score = coverage_ratio(ground_truth, context_text)
        reasoning = f"Heuristic: {score:.0%} of the ground truth's content words appear in the retrieved context."
        return MetricResult(self.key, self.label, score, reasoning, method="heuristic")
