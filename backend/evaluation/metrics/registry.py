"""
Single source of truth for "what metrics does LLM Evaluation compute".
Both `backend/evaluation/runner.py` and the Explainability panel in
`app/pages/llm_evaluation.py` iterate this list rather than
hardcoding metric classes, so adding a new metric later is a
one-line change here.
"""

from __future__ import annotations

from backend.evaluation.metrics.answer_correctness import AnswerCorrectnessMetric
from backend.evaluation.metrics.answer_relevancy import AnswerRelevancyMetric
from backend.evaluation.metrics.base import BaseMetric
from backend.evaluation.metrics.context_precision import ContextPrecisionMetric
from backend.evaluation.metrics.context_recall import ContextRecallMetric
from backend.evaluation.metrics.faithfulness import FaithfulnessMetric
from backend.evaluation.metrics.hallucination import HallucinationMetric


def get_all_metrics() -> list[BaseMetric]:
    """Return one fresh instance of every registered metric."""
    return [
        FaithfulnessMetric(),
        AnswerRelevancyMetric(),
        ContextPrecisionMetric(),
        ContextRecallMetric(),
        HallucinationMetric(),
        AnswerCorrectnessMetric(),
    ]
