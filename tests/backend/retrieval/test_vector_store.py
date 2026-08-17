import numpy as np

from backend.retrieval.vector_store import InMemoryVectorStore, get_vector_store


def _unit(v):
    v = np.array(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_in_memory_store_ranks_by_cosine_similarity():
    store = InMemoryVectorStore()
    embeddings = np.array([_unit([1.0, 0.0]), _unit([0.9, 0.436]), _unit([0.0, 1.0])])
    store.build(["a", "b", "c"], embeddings)

    results = store.search(_unit([1.0, 0.0]), top_k=3)
    ids = [r[0] for r in results]
    assert ids == ["a", "b", "c"]
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_in_memory_store_respects_top_k():
    store = InMemoryVectorStore()
    embeddings = np.array([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])])
    store.build(["a", "b", "c"], embeddings)
    results = store.search(_unit([1, 0]), top_k=2)
    assert len(results) == 2


def test_in_memory_store_empty_before_build():
    store = InMemoryVectorStore()
    assert store.search(_unit([1, 0]), top_k=3) == []


def test_get_vector_store_default_is_in_memory():
    store = get_vector_store()
    assert store.name == "in_memory"


def test_get_vector_store_falls_back_when_backend_unavailable():
    # faiss-cpu/chromadb are listed in requirements.txt but may not be
    # installed in every environment (e.g. this offline test sandbox).
    # Either a real backend or a graceful in-memory fallback is correct —
    # what must never happen is an exception reaching the caller.
    faiss_store = get_vector_store("faiss")
    chroma_store = get_vector_store("chroma")
    assert faiss_store.name in ("faiss", "in_memory")
    assert chroma_store.name in ("chroma", "in_memory")
