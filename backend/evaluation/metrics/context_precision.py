"""
Context Precision — of everything the retriever handed to the
generator, how much of it was actually relevant to the question.
"""

from __future__ import annotations

from backend.evaluation.metrics.base import BaseMetric, MetricExplainer, MetricResult
from backend.evaluation.metrics.judge import call_judge
from backend.evaluation.metrics.text_utils import token_overlap_f1


class ContextPrecisionMetric(BaseMetric):
    key = "context_precision"
    label = "Context Precision"
    higher_is_better = True
    explainer = MetricExplainer(
        what="The proportion of retrieved context that is actually relevant to answering the question.",
        why="Irrelevant retrieved context wastes token budget, increases cost and latency, and can distract "
        "the generator into producing worse or off-topic answers — even when a relevant chunk was also "
        "retrieved alongside the noise.",
        how="An LLM judge reads the question and each retrieved chunk and scores what fraction of the "
        "retrieved context is actually relevant. Offline, a token-overlap heuristic compares the question's "
        "vocabulary against the retrieved context as a rough substitute.",
    )

    def compute(self, question: str, answer: str, contexts: list[str], ground_truth: str, client) -> MetricResult:
        context_text = "\n\n".join(contexts)
        if not context_text.strip():
            return MetricResult(self.key, self.label, 0.0, "No context was retrieved for this question.", method="heuristic")

        judged = call_judge(
            client,
            f"Question:\n\"\"\"\n{question}\n\"\"\"\n\n"
            f"Retrieved context:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            "Score what fraction of the retrieved context is actually relevant/necessary for answering the "
            "question: 1.0 means all of it is relevant, 0.0 means none of it is.",
        )
        if judged is not None:
            score, reasoning = judged
            return MetricResult(self.key, self.label, score, reasoning, method="llm_judge")

        score = token_overlap_f1(question, context_text)
        reasoning = f"Heuristic: question/context token overlap score of {score:.2f}."
        return MetricResult(self.key, self.label, score, reasoning, method="heuristic")
