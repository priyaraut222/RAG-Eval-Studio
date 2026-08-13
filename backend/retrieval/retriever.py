"""
Retriever implementations used to actually fetch chunks for a query,
so Retrieval Evaluation and the Experiment Manager have something to
score without requiring the user to bring a live external pipeline.

Two implementations:
- `TfidfRetriever` — pure numpy, no model download. This is the
  default so the app is fully demoable offline; it's also a
  legitimate, fast retrieval baseline in its own right.
- `EmbeddingRetriever` — wraps `sentence-transformers` when the
  package/model is available, for a closer approximation of a real
  semantic-search RAG pipeline.

Both implement the same `BaseRetriever` interface so the rest of the
app (metrics, experiment runner) never needs to know which is active.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

import numpy as np

from backend.dataset.chunking import Chunk
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class RetrievedChunk:
    """One scored result from `BaseRetriever.retrieve()`."""

    chunk_id: str
    text: str
    score: float


class BaseRetriever(ABC):
    """Common interface every retriever implementation satisfies."""

    name: str

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Build the searchable index from a list of source chunks."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks for `query`, best match first."""
        raise NotImplementedError


class TfidfRetriever(BaseRetriever):
    """TF-IDF + cosine similarity, implemented directly on numpy.

    No sklearn/model-download dependency, so this is the default
    retriever — it lets the whole evaluation loop run in an
    environment with no internet access and no API keys.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._vocab: dict[str, int] = {}
        self._doc_vectors: np.ndarray | None = None
        self._idf: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [f"chunk_{c.chunk_index}" for c in chunks]
        self._chunk_texts = [c.text for c in chunks]

        tokenized_docs = [_tokenize(text) for text in self._chunk_texts]
        vocab: dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocab = vocab

        n_docs = len(tokenized_docs)
        n_vocab = len(vocab)
        term_freq = np.zeros((n_docs, n_vocab), dtype=np.float64)
        doc_freq = np.zeros(n_vocab, dtype=np.float64)

        for doc_idx, tokens in enumerate(tokenized_docs):
            counts = Counter(tokens)
            total = max(len(tokens), 1)
            for token, count in counts.items():
                col = vocab[token]
                term_freq[doc_idx, col] = count / total
                doc_freq[col] += 1

        idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1  # smoothed idf, avoids div-by-zero
        self._idf = idf
        self._doc_vectors = term_freq * idf  # broadcast over columns

    def _vectorize_query(self, query: str) -> np.ndarray:
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        tokens = _tokenize(query)
        if not tokens:
            return vec
        counts = Counter(tokens)
        total = len(tokens)
        for token, count in counts.items():
            col = self._vocab.get(token)
            if col is not None:
                vec[col] = (count / total) * self._idf[col]
        return vec

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._doc_vectors is None or len(self._chunk_ids) == 0:
            return []

        query_vec = self._vectorize_query(query)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        doc_norms = np.linalg.norm(self._doc_vectors, axis=1)
        doc_norms[doc_norms == 0] = 1e-12  # avoid div-by-zero for empty chunks

        similarities = (self._doc_vectors @ query_vec) / (doc_norms * query_norm)
        top_indices = np.argsort(-similarities)[:top_k]

        return [
            RetrievedChunk(chunk_id=self._chunk_ids[i], text=self._chunk_texts[i], score=float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0
        ]


class EmbeddingRetriever(BaseRetriever):
    """Semantic retriever backed by a `sentence-transformers` model.

    Construction raises `ImportError`/`OSError` if the package or
    model weights aren't available (e.g. no network to download
    them) — callers should catch that and fall back to
    `TfidfRetriever`. See `get_retriever()` below, which does exactly
    that.
    """

    name = "embedding"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, deferred

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._doc_embeddings: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [f"chunk_{c.chunk_index}" for c in chunks]
        self._chunk_texts = [c.text for c in chunks]
        self._doc_embeddings = self._model.encode(self._chunk_texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._doc_embeddings is None or len(self._chunk_ids) == 0:
            return []
        query_embedding = self._model.encode([query], normalize_embeddings=True)[0]
        similarities = self._doc_embeddings @ query_embedding
        top_indices = np.argsort(-similarities)[:top_k]
        return [
            RetrievedChunk(chunk_id=self._chunk_ids[i], text=self._chunk_texts[i], score=float(similarities[i]))
            for i in top_indices
        ]


def get_retriever(name: str = "tfidf", embedding_model: str | None = None) -> BaseRetriever:
    """Factory: build a retriever, falling back to TF-IDF if embeddings aren't available."""
    if name == "embedding":
        try:
            return EmbeddingRetriever(model_name=embedding_model or "sentence-transformers/all-MiniLM-L6-v2")
        except Exception as exc:  # ImportError, OSError (no network for model download), etc.
            logger.warning(f"Embedding retriever unavailable ({exc}) — falling back to TF-IDF")
            return TfidfRetriever()
    return TfidfRetriever()
