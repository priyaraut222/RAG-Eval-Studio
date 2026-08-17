"""
Answer Correctness — how well the generated answer matches the
ground-truth answer, factually. Distinct from Faithfulness (grounded
in context) and Relevancy (addresses the question) — an answer can
be faithful and relevant while still being factually wrong.
"""

from __future__ import annotations

from backend.evaluation.metrics.base import BaseMetric, MetricExplainer, MetricResult
from backend.evaluation.metrics.judge import call_judge
from backend.evaluation.metrics.text_utils import token_overlap_f1


class AnswerCorrectnessMetric(BaseMetric):
    key = "answer_correctness"
    label = "Answer Correctness"
    higher_is_better = True
    explainer = MetricExplainer(
        what="How factually consistent the generated answer is with the known-correct ground-truth answer.",
        why="This is the closest thing to an 'is it actually right' check. A faithful, relevant answer can "
        "still get the facts wrong if the context itself was ambiguous or the generator misread it — this "
        "metric is the final check against a trusted reference.",
        how="An LLM judge compares the generated answer to the ground truth and scores factual agreement. "
        "Offline, a token-overlap heuristic (similar to an F1 score) approximates lexical/factual overlap "
        "between the two answers.",
    )

    def compute(self, question: str, answer: str, contexts: list[str], ground_truth: str, client) -> MetricResult:
        judged = call_judge(
            client,
            f"Ground truth answer:\n\"\"\"\n{ground_truth}\n\"\"\"\n\n"
            f"Generated answer:\n\"\"\"\n{answer}\n\"\"\"\n\n"
            "Score how factually consistent the generated answer is with the ground truth answer: 1.0 means "
            "they agree on all key facts, 0.0 means they contradict or share nothing in common.",
        )
        if judged is not None:
            score, reasoning = judged
            return MetricResult(self.key, self.label, score, reasoning, method="llm_judge")

        score = token_overlap_f1(ground_truth, answer)
        reasoning = f"Heuristic: ground-truth/answer token overlap score of {score:.2f}."
        return MetricResult(self.key, self.label, score, reasoning, method="heuristic")
