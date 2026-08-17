"""
Hallucination — the inverse framing of faithfulness: how much of the
generated answer appears to be fabricated or drawn from outside the
retrieved context, rather than supported by it. Scored so that HIGHER
means MORE hallucination (worse), unlike every other metric here.
"""

from __future__ import annotations

from backend.evaluation.metrics.base import BaseMetric, MetricExplainer, MetricResult
from backend.evaluation.metrics.judge import call_judge
from backend.evaluation.metrics.text_utils import coverage_ratio


class HallucinationMetric(BaseMetric):
    key = "hallucination"
    label = "Hallucination"
    higher_is_better = False  # a HIGH score here is bad, unlike the other metrics
    explainer = MetricExplainer(
        what="The proportion of the generated answer that appears to be fabricated or unsupported by the "
        "retrieved context — the mirror image of Faithfulness.",
        why="Hallucination is the single most damaging RAG failure mode for user trust: a confident, "
        "fluent answer that's simply not true. Tracking it explicitly (rather than only its inverse, "
        "Faithfulness) makes it a first-class citizen a team can set an alert threshold on.",
        how="An LLM judge identifies claims in the answer that are NOT supported by the retrieved context and "
        "scores what fraction of the answer they represent. Offline, this is estimated as one minus the "
        "token-coverage heuristic used for Faithfulness.",
    )

    def compute(self, question: str, answer: str, contexts: list[str], ground_truth: str, client) -> MetricResult:
        context_text = "\n\n".join(contexts)

        judged = call_judge(
            client,
            f"Context:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            f"Answer:\n\"\"\"\n{answer}\n\"\"\"\n\n"
            "Score how much of the answer is fabricated or NOT supported by the context: 1.0 means the answer "
            "is almost entirely unsupported/hallucinated, 0.0 means every claim is backed by the context.",
        )
        if judged is not None:
            score, reasoning = judged
            return MetricResult(self.key, self.label, score, reasoning, method="llm_judge")

        grounded = coverage_ratio(answer, context_text)
        score = 1.0 - grounded
        reasoning = f"Heuristic: an estimated {score:.0%} of the answer's content words do NOT appear in the retrieved context."
        return MetricResult(self.key, self.label, score, reasoning, method="heuristic")
