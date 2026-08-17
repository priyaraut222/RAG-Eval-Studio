from backend.evaluation.metrics.answer_correctness import AnswerCorrectnessMetric
from backend.evaluation.metrics.answer_relevancy import AnswerRelevancyMetric
from backend.evaluation.metrics.context_precision import ContextPrecisionMetric
from backend.evaluation.metrics.context_recall import ContextRecallMetric
from backend.evaluation.metrics.faithfulness import FaithfulnessMetric
from backend.evaluation.metrics.hallucination import HallucinationMetric
from backend.evaluation.metrics.registry import get_all_metrics


class _LocalClient:
    """Forces every metric onto its heuristic path (no LLM judge call)."""

    active_provider = "local"


QUESTION = "Who designed the Eiffel Tower?"
CONTEXT = "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889 for the World Fair in Paris."
GROUND_TRUTH = "Gustave Eiffel designed the Eiffel Tower."
GOOD_ANSWER = "Gustave Eiffel designed the Eiffel Tower, completed in 1889."
UNRELATED_ANSWER = "Bananas are a good source of potassium."


def test_faithfulness_high_when_answer_grounded_in_context():
    result = FaithfulnessMetric().compute(
        question=QUESTION, answer=CONTEXT, contexts=[CONTEXT], ground_truth=GROUND_TRUTH, client=_LocalClient()
    )
    assert result.score == 1.0
    assert result.method == "heuristic"


def test_faithfulness_low_when_answer_unrelated_to_context():
    result = FaithfulnessMetric().compute(
        question=QUESTION, answer=UNRELATED_ANSWER, contexts=[CONTEXT], ground_truth=GROUND_TRUTH, client=_LocalClient()
    )
    assert result.score < 0.2


def test_hallucination_is_inverse_of_faithfulness():
    faithfulness = FaithfulnessMetric().compute(
        question=QUESTION, answer=GOOD_ANSWER, contexts=[CONTEXT], ground_truth=GROUND_TRUTH, client=_LocalClient()
    )
    hallucination = HallucinationMetric().compute(
        question=QUESTION, answer=GOOD_ANSWER, contexts=[CONTEXT], ground_truth=GROUND_TRUTH, client=_LocalClient()
    )
    assert abs((1 - faithfulness.score) - hallucination.score) < 1e-9


def test_hallucination_higher_is_better_is_false():
    assert HallucinationMetric().higher_is_better is False
    assert FaithfulnessMetric().higher_is_better is True


def test_context_precision_zero_when_no_context():
    result = ContextPrecisionMetric().compute(
        question=QUESTION, answer=GOOD_ANSWER, contexts=[""], ground_truth=GROUND_TRUTH, client=_LocalClient()
    )
    assert result.score == 0.0


def test_context_recall_high_when_ground_truth_covered_by_context():
    result = ContextRecallMetric().compute(
        question=QUESTION, answer=GOOD_ANSWER, contexts=[CONTEXT], ground_truth=GROUND_TRUTH, client=_LocalClient()
    )
    assert result.score > 0.5


def test_answer_correctness_high_for_matching_answer():
    result = AnswerCorrectnessMetric().compute(
        question=QUESTION, answer=GOOD_ANSWER, contexts=[CONTEXT], ground_truth=GROUND_TRUTH, client=_LocalClient()
    )
    assert result.score > 0.5


def test_answer_correctness_low_for_unrelated_answer():
    result = AnswerCorrectnessMetric().compute(
        question=QUESTION, answer=UNRELATED_ANSWER, contexts=[CONTEXT], ground_truth=GROUND_TRUTH, client=_LocalClient()
    )
    assert result.score < 0.2


def test_answer_relevancy_low_for_off_topic_answer():
    result = AnswerRelevancyMetric().compute(
        question=QUESTION, answer=UNRELATED_ANSWER, contexts=[CONTEXT], ground_truth=GROUND_TRUTH, client=_LocalClient()
    )
    assert result.score < 0.2


def test_all_metric_scores_are_bounded_zero_to_one():
    for metric in get_all_metrics():
        result = metric.compute(
            question=QUESTION, answer=GOOD_ANSWER, contexts=[CONTEXT], ground_truth=GROUND_TRUTH, client=_LocalClient()
        )
        assert 0.0 <= result.score <= 1.0, f"{metric.key} produced out-of-range score {result.score}"


def test_registry_returns_six_distinct_metrics():
    metrics = get_all_metrics()
    keys = {m.key for m in metrics}
    assert len(keys) == 6
    assert keys == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "hallucination",
        "answer_correctness",
    }


def test_every_metric_has_explainability_copy():
    for metric in get_all_metrics():
        assert metric.explainer.what.strip()
        assert metric.explainer.why.strip()
        assert metric.explainer.how.strip()
