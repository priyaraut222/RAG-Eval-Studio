"""
Shared types every LLM evaluation metric implements.

Each concrete metric (faithfulness.py, hallucination.py, ...) is a
small class satisfying `BaseMetric`: it knows how to score one item
via an LLM judge, falls back to a heuristic when no judge is
available, and carries its own `MetricExplainer` so the UI's
Explainability panel never has to hardcode metric descriptions
elsewhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricExplainer:
    """What/why/how copy shown in the Explainability panel for one metric."""

    what: str
    why: str
    how: str


@dataclass
class MetricResult:
    """The outcome of scoring one dataset item on one metric."""

    key: str
    label: str
    score: float  # normalized to [0, 1]
    reasoning: str
    method: str  # "llm_judge" or "heuristic" — which path actually produced the score


class BaseMetric(ABC):
    """Common interface for every LLM evaluation metric."""

    key: str
    label: str
    higher_is_better: bool = True
    explainer: MetricExplainer

    @abstractmethod
    def compute(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
        client,
    ) -> MetricResult:
        """Score one item. `client` is an `LLMClient`; metrics decide
        internally whether to use its LLM-judge path or their heuristic
        fallback (typically based on `client.active_provider`)."""
        raise NotImplementedError
