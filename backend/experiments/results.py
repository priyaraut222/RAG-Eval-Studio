"""
`ExperimentRun` — the persisted result of running one `ExperimentConfig`
against one dataset: every item's retrieved context, generated answer,
retrieval metrics, LLM metrics, and cost/latency, plus dataset-level
aggregates. Pydantic (not a plain dataclass) so it round-trips to JSON
cleanly via `backend/experiments/storage.py`, the same pattern as
`EvalDataset`.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from backend.experiments.config import ExperimentConfig


class ItemResult(BaseModel):
    """One dataset item's outcome under one experiment configuration."""

    item_id: str
    question: str
    ground_truth: str
    retrieved_context: str
    generated_answer: str

    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    llm_scores: dict[str, float] = Field(default_factory=dict)
    llm_reasoning: dict[str, str] = Field(default_factory=dict)

    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0

    def overall_score(self) -> float:
        """Mean of the "higher is better" LLM metrics (Hallucination excluded)."""
        scores = [v for k, v in self.llm_scores.items() if k != "hallucination"]
        return sum(scores) / len(scores) if scores else 0.0


class ExperimentRun(BaseModel):
    """Full result of running one `ExperimentConfig` against one dataset."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    config: ExperimentConfig
    dataset_id: str
    dataset_name: str

    items: list[ItemResult] = Field(default_factory=list)
    aggregate_retrieval: dict[str, float] = Field(default_factory=dict)
    aggregate_llm: dict[str, float] = Field(default_factory=dict)

    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def overall_score(self) -> float:
        """Blended score across retrieval + LLM aggregates (Hallucination excluded).

        This is the single number the comparison table sorts "best
        configuration" by.
        """
        parts = list(self.aggregate_retrieval.values())
        parts += [v for k, v in self.aggregate_llm.items() if k != "hallucination"]
        return sum(parts) / len(parts) if parts else 0.0

    @property
    def worst_items(self) -> list[ItemResult]:
        return sorted(self.items, key=lambda r: r.overall_score())

    @property
    def best_items(self) -> list[ItemResult]:
        return sorted(self.items, key=lambda r: r.overall_score(), reverse=True)

    def to_json_file(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path) -> "ExperimentRun":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
