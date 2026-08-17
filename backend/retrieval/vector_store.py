"""
Vector store backends for `EmbeddingRetriever`.

This is what makes "compare different vector databases" a real axis
in the Experiment Manager rather than a cosmetic label: each backend
actually indexes and searches embeddings differently.

- `InMemoryVectorStore` — plain numpy cosine similarity. Always
  available, zero dependencies, the default.
- `FaissVectorStore` — wraps `faiss.IndexFlatIP` over normalized
  embeddings (inner product == cosine similarity when normalized).
- `ChromaVectorStore` — wraps an ephemeral `chromadb` collection.

Both `FaissVectorStore` and `ChromaVectorStore` raise `ImportError`
at construction if their package isn't installed; `get_vector_store()`
catches that and falls back to `InMemoryVectorStore` with a logged
warning, matching the same graceful-degradation pattern used by
`EmbeddingRetriever` itself for missing `sentence-transformers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class BaseVectorStore(ABC):
    """Common interface every vector store backend satisfies."""

    name: str

    @abstractmethod
    def build(self, ids: list[str], embeddings: np.ndarray) -> None:
        """Index `embeddings` (already L2-normalized), keyed by `ids`."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return up to `top_k` (id, similarity_score) pairs, best first."""
        raise NotImplementedError


class InMemoryVectorStore(BaseVectorStore):
    """Numpy-only cosine similarity search — no external dependency."""

    name = "in_memory"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    def build(self, ids: list[str], embeddings: np.ndarray) -> None:
        self._ids = ids
        self._embeddings = embeddings

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._embeddings is None or len(self._ids) == 0:
            return []
        similarities = self._embeddings @ query_embedding
        top_indices = np.argsort(-similarities)[:top_k]
        return [(self._ids[i], float(similarities[i])) for i in top_indices]


class FaissVectorStore(BaseVectorStore):
    """FAISS `IndexFlatIP` over normalized embeddings (exact search)."""

    name = "faiss"

    def __init__(self) -> None:
        import faiss  # deferred — raises ImportError if not installed

        self._faiss = faiss
        self._index = None
        self._ids: list[str] = []

    def build(self, ids: list[str], embeddings: np.ndarray) -> None:
        self._ids = ids
        dim = embeddings.shape[1]
        self._index = self._faiss.IndexFlatIP(dim)
        self._index.add(np.ascontiguousarray(embeddings.astype("float32")))

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        query = np.ascontiguousarray(query_embedding.astype("float32")).reshape(1, -1)
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))
        return [(self._ids[i], float(s)) for s, i in zip(scores[0], indices[0]) if i != -1]


class ChromaVectorStore(BaseVectorStore):
    """Ephemeral (in-process) ChromaDB collection.

    Embeddings are supplied directly (from the same encoder the rest
    of the app uses) rather than letting Chroma compute its own, so
    results are comparable across vector-store backends for the same
    retriever configuration.
    """

    name = "chroma"

    def __init__(self) -> None:
        import chromadb  # deferred — raises ImportError if not installed

        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(name="rag_eval_studio", get_or_create=True)

    def build(self, ids: list[str], embeddings: np.ndarray) -> None:
        if not ids:
            return
        # Chroma collections are append-only per id; clear any previous state
        # from a prior `build()` call on this instance before re-adding.
        try:
            existing = self._collection.get()
            if existing and existing.get("ids"):
                self._collection.delete(ids=existing["ids"])
        except Exception:  # pragma: no cover — best-effort reset
            pass
        self._collection.add(ids=ids, embeddings=embeddings.tolist())

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        result = self._collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        # Chroma returns distance (lower = closer); convert to a similarity-style score.
        return [(doc_id, 1.0 - dist) for doc_id, dist in zip(ids, distances)]


def get_vector_store(name: str = "in_memory") -> BaseVectorStore:
    """Factory: build the requested vector store, falling back to in-memory if unavailable."""
    if name == "faiss":
        try:
            return FaissVectorStore()
        except Exception as exc:  # ImportError if faiss-cpu isn't installed
            logger.warning(f"FAISS vector store unavailable ({exc}) — falling back to in-memory")
            return InMemoryVectorStore()
    if name == "chroma":
        try:
            return ChromaVectorStore()
        except Exception as exc:  # ImportError if chromadb isn't installed
            logger.warning(f"Chroma vector store unavailable ({exc}) — falling back to in-memory")
            return InMemoryVectorStore()
    return InMemoryVectorStore()
