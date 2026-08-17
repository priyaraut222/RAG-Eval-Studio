"""
Tests the heuristic (offline) path of every metric — the path that
runs with no API key configured. The LLM-judge path is exercised
indirectly via `backend/evaluation/runner.py`'s integration test and
isn't re-tested here since it just delegates to a real provider call.
"""

from backend.evaluation.metrics.answer_correctness import AnswerCorrectnessMetric
from backend.evaluation.metrics.answer_relevancy import AnswerRelevancyMetric
from backend.evaluation.metrics.context_precision import ContextPrecisionMetric
from backend.evaluation.metrics.context_recall import ContextRecallMetric
from backend.evaluation.metrics.faithfulness import FaithfulnessMetric
from backend.evaluation.metrics.hallucination import HallucinationMetric
from backend.evaluation.metrics.registry import get_all_metrics


class _FakeLocalClient:
    """Stands in for `LLMClient` on the local/offline provider — every
    metric's `call_judge()` should see this and skip straight to its
    heuristic, without needing real settings/pydantic wired up."""

    active_provider = "local"


QUESTION = "Who designed the Eiffel Tower?"
CONTEXT = "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889."
GROUND_TRUTH = "Gustave Eiffel"
GOOD_ANSWER = "The Eiffel Tower was designed by Gustave Eiffel."
BAD_ANSWER = "Quantum entanglement enables faster-than-light communication."


def test_get_all_metrics_returns_six_metrics():
    metrics = get_all_metrics()
    assert len(metrics) == 6
    keys = {m.key for m in metrics}
    assert keys == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "hallucination",
        "answer_correctness",
    }


def test_every_metric_has_explainer_copy():
    for metric in get_all_metrics():
        assert metric.explainer.what
        assert metric.explainer.why
        assert metric.explainer.how


def test_faithfulness_scores_grounded_answer_higher_than_ungrounded():
    metric = FaithfulnessMetric()
    good = metric.compute(QUESTION, GOOD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
    bad = metric.compute(QUESTION, BAD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
    assert good.method == "heuristic"
    assert good.score > bad.score


def test_hallucination_is_inverse_of_faithfulness_heuristic():
    faithfulness = FaithfulnessMetric().compute(QUESTION, GOOD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
    hallucination = HallucinationMetric().compute(QUESTION, GOOD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
    assert abs((1 - faithfulness.score) - hallucination.score) < 1e-9


def test_hallucination_higher_is_better_is_false():
    assert HallucinationMetric().higher_is_better is False


def test_answer_relevancy_scores_on_topic_answer_higher():
    metric = AnswerRelevancyMetric()
    on_topic = metric.compute(QUESTION, GOOD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
    off_topic = metric.compute(QUESTION, BAD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
    assert on_topic.score > off_topic.score


def test_context_precision_empty_context_scores_zero():
    metric = ContextPrecisionMetric()
    result = metric.compute(QUESTION, GOOD_ANSWER, [""], GROUND_TRUTH, _FakeLocalClient())
    assert result.score == 0.0


def test_context_recall_full_ground_truth_coverage():
    metric = ContextRecallMetric()
    result = metric.compute(QUESTION, GOOD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
    assert result.score == 1.0  # "Gustave Eiffel" is fully present in CONTEXT


def test_answer_correctness_matches_ground_truth_higher_than_unrelated():
    metric = AnswerCorrectnessMetric()
    correct = metric.compute(QUESTION, GOOD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
    incorrect = metric.compute(QUESTION, BAD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
    assert correct.score > incorrect.score


def test_all_metrics_produce_scores_in_valid_range():
    for metric in get_all_metrics():
        result = metric.compute(QUESTION, GOOD_ANSWER, [CONTEXT], GROUND_TRUTH, _FakeLocalClient())
        assert 0.0 <= result.score <= 1.0
        assert result.method == "heuristic"
        assert result.reasoning
