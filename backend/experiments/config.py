"""
`ExperimentConfig` — one full RAG pipeline configuration to run and
compare: chunking, retriever, embedding model, vector store, Top-K,
reranker on/off, and LLM provider. This is the unit of comparison in
the Experiment Manager, and the unit of persistence in
`backend/experiments/storage.py`.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

RetrieverName = Literal["tfidf", "embedding"]
VectorStoreName = Literal["in_memory", "faiss", "chroma"]
LLMProvider = Literal["openai", "gemini", "local"]

EMBEDDING_MODEL_CHOICES: list[str] = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
]


class ExperimentConfig(BaseModel):
    """One named configuration of every axis the platform lets you vary."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str

    chunk_size: int = 800
    chunk_overlap: int = 100

    retriever: RetrieverName = "tfidf"
    embedding_model: str = EMBEDDING_MODEL_CHOICES[0]
    vector_store: VectorStoreName = "in_memory"
    top_k: int = 5
    use_reranker: bool = False

    llm_provider: LLMProvider = "local"

    def summary(self) -> str:
        """One-line, comparison-table-friendly description of this config."""
        parts = [
            f"chunk={self.chunk_size}/{self.chunk_overlap}",
            f"retriever={self.retriever}",
        ]
        if self.retriever == "embedding":
            parts.append(f"vs={self.vector_store}")
        parts.append(f"top_k={self.top_k}")
        if self.use_reranker:
            parts.append("+reranker")
        parts.append(f"llm={self.llm_provider}")
        return " · ".join(parts)
