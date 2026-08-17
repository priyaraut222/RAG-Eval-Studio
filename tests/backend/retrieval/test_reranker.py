from backend.retrieval.reranker import rerank
from backend.retrieval.retriever import RetrievedChunk


def test_rerank_reorders_by_query_overlap():
    candidates = [
        RetrievedChunk(chunk_id="a", text="Bananas are rich in potassium.", score=0.9),
        RetrievedChunk(chunk_id="b", text="The Eiffel Tower was designed by Gustave Eiffel.", score=0.1),
    ]
    # candidate "a" has the higher original retriever score, but "b" is the
    # actual match for this query — a working reranker must flip the order.
    result = rerank("Who designed the Eiffel Tower?", candidates, top_k=2)
    assert result[0].chunk_id == "b"


def test_rerank_respects_top_k():
    candidates = [
        RetrievedChunk(chunk_id=str(i), text=f"chunk number {i} about topic {i}", score=0.0) for i in range(10)
    ]
    result = rerank("topic 5", candidates, top_k=3)
    assert len(result) == 3


def test_rerank_empty_candidates_returns_empty():
    assert rerank("anything", [], top_k=5) == []
