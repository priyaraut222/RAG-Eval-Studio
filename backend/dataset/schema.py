"""
Core data model for evaluation datasets.

Every dataset the app works with — whether built manually, parsed
from an upload, or LLM-synthesized — is a list of `EvalItem`s. This
schema is the contract between the Dataset Builder, Retrieval
Evaluation, LLM Evaluation, and Reports modules, so it lives in
`backend/dataset/` rather than any one feature.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ItemSource = Literal["manual", "synthetic", "imported"]


class EvalItem(BaseModel):
    """One evaluation example: a question with its expected answer/context.

    `expected_context` is the full passage(s) a good retriever should
    surface; `expected_chunk` is the specific chunk-sized excerpt that
    most directly answers the question (useful once documents are
    split for retrieval — see `backend/dataset/chunking.py`).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    question: str
    ground_truth: str
    expected_context: str = ""
    expected_chunk: str = ""
    source_document: str = ""
    source: ItemSource = "manual"

    def is_complete(self) -> bool:
        """Whether this item has the minimum fields needed for evaluation."""
        return bool(self.question.strip() and self.ground_truth.strip())


class EvalDataset(BaseModel):
    """A named, ordered collection of `EvalItem`s plus light metadata."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    description: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    items: list[EvalItem] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.items)

    def add_item(self, item: EvalItem) -> None:
        self.items.append(item)

    def to_json_file(self, path: Path) -> None:
        """Serialize this dataset to a JSON file, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: Path) -> "EvalDataset":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
