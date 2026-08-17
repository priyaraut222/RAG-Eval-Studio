import csv
import io
import json

from backend.dataset.schema import EvalDataset, EvalItem
from backend.experiments.config import ExperimentConfig
from backend.experiments.runner import run_experiment
from backend.reports.generator import (
    generate_comparison_csv,
    generate_comparison_json,
    generate_comparison_markdown,
    generate_csv,
    generate_json,
    generate_markdown,
)

SOURCE = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was designed by Gustave Eiffel and completed in 1889 for the World Fair. "
    "The Great Wall of China stretches over 13000 miles across northern China."
)


def _dataset() -> EvalDataset:
    return EvalDataset(
        name="landmarks",
        items=[
            EvalItem(
                question="Who designed the Eiffel Tower?",
                ground_truth="Gustave Eiffel.",
                expected_chunk="It was designed by Gustave Eiffel and completed in 1889 for the World Fair.",
            ),
            EvalItem(
                question="How long is the Great Wall?",
                ground_truth="Over 13000 miles.",
                expected_chunk="The Great Wall of China stretches over 13000 miles across northern China.",
            ),
        ],
    )


def _run(name: str = "baseline", **overrides) -> "ExperimentRun":
    fields = {"chunk_size": 100, "chunk_overlap": 15, "top_k": 3}
    fields.update(overrides)
    config = ExperimentConfig(name=name, **fields)
    return run_experiment(_dataset(), config, SOURCE)


def test_generate_csv_has_one_row_per_item_and_all_metric_columns():
    run = _run()
    rows = list(csv.DictReader(io.StringIO(generate_csv(run))))
    assert len(rows) == len(run.items)
    for key in ("precision_at_k", "faithfulness", "answer_correctness", "latency_ms", "estimated_cost"):
        assert key in rows[0]


def test_generate_json_round_trips_config_and_items():
    run = _run()
    parsed = json.loads(generate_json(run))
    assert parsed["config"]["name"] == run.config.name
    assert len(parsed["items"]) == len(run.items)


def test_generate_markdown_contains_key_sections():
    run = _run()
    md = generate_markdown(run)
    for heading in ("# Evaluation Report", "## Retrieval Metrics", "## LLM Evaluation Metrics", "## Best Performing Questions", "## Worst Performing Questions"):
        assert heading in md
    assert run.config.name in md


def test_generate_markdown_includes_every_question():
    run = _run()
    md = generate_markdown(run)
    for item in run.items:
        assert item.question in md


def test_comparison_csv_has_one_row_per_run():
    run_a = _run("baseline")
    run_b = _run("reranked", use_reranker=True, top_k=2)
    rows = list(csv.DictReader(io.StringIO(generate_comparison_csv([run_a, run_b]))))
    assert len(rows) == 2
    assert {r["configuration"] for r in rows} == {"baseline", "reranked"}


def test_comparison_json_is_a_list_of_full_runs():
    run_a = _run("baseline")
    run_b = _run("reranked", use_reranker=True, top_k=2)
    parsed = json.loads(generate_comparison_json([run_a, run_b]))
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert {p["config"]["name"] for p in parsed} == {"baseline", "reranked"}


def test_comparison_markdown_names_the_best_configuration():
    run_a = _run("baseline")
    run_b = _run("reranked", use_reranker=True, top_k=2)
    best = max([run_a, run_b], key=lambda r: r.overall_score())
    md = generate_comparison_markdown([run_a, run_b])
    assert "Best overall configuration" in md
    assert best.config.name in md


def test_comparison_markdown_handles_single_run():
    run = _run()
    md = generate_comparison_markdown([run])
    assert run.config.name in md
