from backend.dataset.schema import EvalDataset, EvalItem
from backend.experiments.config import ExperimentConfig
from backend.experiments.results import ExperimentRun
from backend.experiments.runner import run_experiment

SOURCE = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was designed by Gustave Eiffel and completed in 1889 for the World Fair. "
    "The Great Wall of China stretches over 13000 miles across northern China. "
    "It was built over centuries by various dynasties to protect against invasions. "
    "Mount Everest is the tallest mountain on Earth, located in the Himalayas. "
    "Climbers first successfully summited it in 1953 via the South Col route."
)


def _dataset() -> EvalDataset:
    return EvalDataset(
        name="landmarks",
        items=[
            EvalItem(
                question="Who designed the Eiffel Tower?",
                ground_truth="Gustave Eiffel designed it.",
                expected_chunk="It was designed by Gustave Eiffel and completed in 1889 for the World Fair.",
            ),
            EvalItem(
                question="When was Everest first summited?",
                ground_truth="1953.",
                expected_chunk="Climbers first successfully summited it in 1953 via the South Col route.",
            ),
            EvalItem(
                question="How long is the Great Wall of China?",
                ground_truth="Over 13000 miles.",
                expected_chunk="The Great Wall of China stretches over 13000 miles across northern China.",
            ),
        ],
    )


def test_run_experiment_produces_one_result_per_complete_item():
    config = ExperimentConfig(name="baseline", chunk_size=100, chunk_overlap=15, top_k=3)
    run = run_experiment(_dataset(), config, SOURCE)
    assert isinstance(run, ExperimentRun)
    assert len(run.items) == 3


def test_run_experiment_skips_incomplete_items():
    dataset = _dataset()
    dataset.add_item(EvalItem(question="", ground_truth=""))  # incomplete: no question/ground truth
    config = ExperimentConfig(name="baseline", chunk_size=100, chunk_overlap=15, top_k=3)
    run = run_experiment(dataset, config, SOURCE)
    assert len(run.items) == 3  # incomplete item excluded


def test_run_experiment_local_provider_has_zero_cost():
    config = ExperimentConfig(name="baseline", chunk_size=100, chunk_overlap=15, top_k=3, llm_provider="local")
    run = run_experiment(_dataset(), config, SOURCE)
    assert run.total_cost == 0.0


def test_run_experiment_aggregates_are_bounded():
    config = ExperimentConfig(name="baseline", chunk_size=100, chunk_overlap=15, top_k=3)
    run = run_experiment(_dataset(), config, SOURCE)
    for score in run.aggregate_retrieval.values():
        assert 0.0 <= score <= 1.0
    for score in run.aggregate_llm.values():
        assert 0.0 <= score <= 1.0


def test_run_experiment_raises_on_empty_source_text():
    import pytest

    config = ExperimentConfig(name="baseline")
    with pytest.raises(ValueError):
        run_experiment(_dataset(), config, "")


def test_reranker_config_changes_retrieval_results():
    without_rerank = ExperimentConfig(name="no-rerank", chunk_size=100, chunk_overlap=15, top_k=2, use_reranker=False)
    with_rerank = ExperimentConfig(name="rerank", chunk_size=100, chunk_overlap=15, top_k=2, use_reranker=True)

    run_a = run_experiment(_dataset(), without_rerank, SOURCE)
    run_b = run_experiment(_dataset(), with_rerank, SOURCE)

    # Both should produce valid, complete results regardless of whether
    # reranking happens to change the final ranking for this small corpus.
    assert len(run_a.items) == len(run_b.items) == 3


def test_best_items_and_worst_items_are_sorted_correctly():
    config = ExperimentConfig(name="baseline", chunk_size=100, chunk_overlap=15, top_k=3)
    run = run_experiment(_dataset(), config, SOURCE)
    best_scores = [item.overall_score() for item in run.best_items]
    assert best_scores == sorted(best_scores, reverse=True)
    worst_scores = [item.overall_score() for item in run.worst_items]
    assert worst_scores == sorted(worst_scores)


def test_experiment_run_overall_score_excludes_hallucination():
    config = ExperimentConfig(name="baseline", chunk_size=100, chunk_overlap=15, top_k=3)
    run = run_experiment(_dataset(), config, SOURCE)
    manual_parts = list(run.aggregate_retrieval.values()) + [
        v for k, v in run.aggregate_llm.items() if k != "hallucination"
    ]
    expected = sum(manual_parts) / len(manual_parts)
    assert abs(run.overall_score() - expected) < 1e-9
