from backend.dataset.schema import EvalDataset, EvalItem
from backend.experiments.config import ExperimentConfig
from backend.experiments.runner import run_experiment
from backend.experiments.storage import delete_run, list_runs, load_run, save_run

SOURCE = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was designed by Gustave Eiffel and completed in 1889 for the World Fair."
)


def _make_run(name: str = "test-run"):
    dataset = EvalDataset(
        name="tiny",
        items=[
            EvalItem(
                question="Who designed the Eiffel Tower?",
                ground_truth="Gustave Eiffel.",
                expected_chunk="It was designed by Gustave Eiffel and completed in 1889 for the World Fair.",
            ),
        ],
    )
    config = ExperimentConfig(name=name, chunk_size=100, chunk_overlap=15, top_k=2)
    return run_experiment(dataset, config, SOURCE)


def test_save_and_load_round_trip_preserves_config_type():
    run = _make_run()
    try:
        save_run(run)
        loaded = load_run(run.id)
        assert isinstance(loaded.config, ExperimentConfig)
        assert loaded.config.name == run.config.name
        assert len(loaded.items) == len(run.items)
        assert abs(loaded.overall_score() - run.overall_score()) < 1e-9
    finally:
        delete_run(run.id)


def test_list_runs_includes_saved_run():
    run = _make_run("listed-run")
    try:
        save_run(run)
        ids = {r.id for r in list_runs()}
        assert run.id in ids
    finally:
        delete_run(run.id)


def test_delete_run_removes_it():
    run = _make_run("deletable-run")
    save_run(run)
    assert delete_run(run.id) is True
    assert run.id not in {r.id for r in list_runs()}


def test_delete_run_returns_false_for_missing_id():
    assert delete_run("does-not-exist") is False
