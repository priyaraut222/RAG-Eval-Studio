from backend.experiments.config import ExperimentConfig


def test_default_config_is_fully_offline_runnable():
    config = ExperimentConfig(name="baseline")
    assert config.retriever == "tfidf"
    assert config.vector_store == "in_memory"
    assert config.llm_provider == "local"
    assert config.use_reranker is False


def test_summary_includes_reranker_flag_only_when_enabled():
    without = ExperimentConfig(name="a", use_reranker=False)
    with_reranker = ExperimentConfig(name="b", use_reranker=True)
    assert "+reranker" not in without.summary()
    assert "+reranker" in with_reranker.summary()


def test_summary_includes_vector_store_only_for_embedding_retriever():
    tfidf_config = ExperimentConfig(name="a", retriever="tfidf", vector_store="faiss")
    embedding_config = ExperimentConfig(name="b", retriever="embedding", vector_store="faiss")
    assert "vs=" not in tfidf_config.summary()
    assert "vs=faiss" in embedding_config.summary()


def test_each_config_gets_a_unique_id():
    a = ExperimentConfig(name="a")
    b = ExperimentConfig(name="b")
    assert a.id != b.id
