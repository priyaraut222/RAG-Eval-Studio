"""
Answer Relevancy — how directly the generated answer addresses the
question that was actually asked, independent of whether it's
factually correct or grounded in context.
"""

from __future__ import annotations

from backend.evaluation.metrics.base import BaseMetric, MetricExplainer, MetricResult
from backend.evaluation.metrics.judge import call_judge
from backend.evaluation.metrics.text_utils import token_overlap_f1


class AnswerRelevancyMetric(BaseMetric):
    key = "answer_relevancy"
    label = "Answer Relevancy"
    higher_is_better = True
    explainer = MetricExplainer(
        what="How directly the answer addresses the specific question asked.",
        why="An answer can be perfectly faithful to the context and still be unhelpful if it's evasive, "
        "over-broad, or answers a different question than the one asked. This metric catches that failure "
        "mode separately from faithfulness or correctness.",
        how="An LLM judge compares the question and answer and scores how directly the answer addresses what "
        "was asked. Offline, a token-overlap heuristic checks how much vocabulary the answer shares with the "
        "question — a rough but zero-dependency proxy for topical relevance.",
    )

    def compute(self, question: str, answer: str, contexts: list[str], ground_truth: str, client) -> MetricResult:
        judged = call_judge(
            client,
            f"Question:\n\"\"\"\n{question}\n\"\"\"\n\n"
            f"Answer:\n\"\"\"\n{answer}\n\"\"\"\n\n"
            "Score how directly and completely the answer addresses the question: 1.0 means it directly and "
            "fully answers what was asked, 0.0 means it's off-topic, evasive, or answers a different question.",
        )
        if judged is not None:
            score, reasoning = judged
            return MetricResult(self.key, self.label, score, reasoning, method="llm_judge")

        score = token_overlap_f1(question, answer)
        reasoning = f"Heuristic: question/answer token overlap score of {score:.2f}."
        return MetricResult(self.key, self.label, score, reasoning, method="heuristic")
