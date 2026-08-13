from backend.dataset.chunking import chunk_text
from backend.retrieval.retriever import TfidfRetriever


SOURCE = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was designed by Gustave Eiffel and completed in 1889 for the World Fair. "
    "The Great Wall of China stretches over 13000 miles across northern China. "
    "It was built over centuries by various dynasties to protect against invasions. "
    "Mount Everest is the tallest mountain on Earth, located in the Himalayas. "
    "Climbers first successfully summited it in 1953 via the South Col route."
)


def _build_retriever() -> TfidfRetriever:
    chunks = chunk_text(SOURCE, chunk_size=100, chunk_overlap=15)
    retriever = TfidfRetriever()
    retriever.index(chunks)
    return retriever


def test_retrieve_returns_results_for_relevant_query():
    retriever = _build_retriever()
    results = retriever.retrieve("Who designed the Eiffel Tower?", top_k=3)
    assert results
    assert "Eiffel" in results[0].text


def test_retrieve_ranks_most_relevant_chunk_first():
    retriever = _build_retriever()
    results = retriever.retrieve("When did climbers first summit Mount Everest?", top_k=3)
    assert "1953" in results[0].text or "Everest" in results[0].text


def test_retrieve_respects_top_k():
    retriever = _build_retriever()
    results = retriever.retrieve("mountain", top_k=2)
    assert len(results) <= 2


def test_retrieve_empty_query_returns_empty():
    retriever = _build_retriever()
    assert retriever.retrieve("", top_k=3) == []


def test_retrieve_before_index_returns_empty():
    retriever = TfidfRetriever()
    assert retriever.retrieve("anything", top_k=3) == []


def test_scores_are_descending():
    retriever = _build_retriever()
    results = retriever.retrieve("Great Wall of China length", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
