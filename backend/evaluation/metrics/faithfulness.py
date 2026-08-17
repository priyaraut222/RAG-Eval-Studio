"""
Faithfulness — how much of the generated answer is actually supported
by the retrieved context, as opposed to being invented or drawn from
the model's parametric knowledge.
"""

from __future__ import annotations

from backend.evaluation.metrics.base import BaseMetric, MetricExplainer, MetricResult
from backend.evaluation.metrics.judge import call_judge
from backend.evaluation.metrics.text_utils import coverage_ratio


class FaithfulnessMetric(BaseMetric):
    key = "faithfulness"
    label = "Faithfulness"
    higher_is_better = True
    explainer = MetricExplainer(
        what="The proportion of claims in the generated answer that are directly supported by the retrieved context.",
        why="A RAG system's core promise is that answers are grounded in retrieved evidence, not the model's own "
        "possibly-wrong internal knowledge. Low faithfulness means the pipeline is generating content the context "
        "doesn't actually back up — a leading cause of user-visible errors.",
        how="An LLM judge reads the answer and the retrieved context together and scores what fraction of the "
        "answer's claims are directly traceable to the context. Offline (no API key), a token-overlap heuristic "
        "estimates the same idea: what share of the answer's content words also appear in the context.",
    )

    def compute(self, question: str, answer: str, contexts: list[str], ground_truth: str, client) -> MetricResult:
        context_text = "\n\n".join(contexts)

        judged = call_judge(
            client,
            f"Context:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            f"Answer:\n\"\"\"\n{answer}\n\"\"\"\n\n"
            "Score how faithful the answer is to the context: 1.0 means every claim in the answer is directly "
            "supported by the context, 0.0 means none of it is. Penalize any claim not backed by the context, "
            "even if the claim happens to be true in general.",
        )
        if judged is not None:
            score, reasoning = judged
            return MetricResult(self.key, self.label, score, reasoning, method="llm_judge")

        score = coverage_ratio(answer, context_text)
        reasoning = f"Heuristic: {score:.0%} of the answer's content words appear in the retrieved context."
        return MetricResult(self.key, self.label, score, reasoning, method="heuristic")
