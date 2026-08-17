"""
Persistence for `ExperimentRun`s — one JSON file per run under
`data/experiments/<id>.json`, mirroring `backend/dataset/storage.py`.
This is what lets the Experiment Manager's "Saved Experiments" tab
(and, later, Reports) compare configurations across sessions rather
than only within one Streamlit run.
"""

from __future__ import annotations

from backend.config.settings import get_settings
from backend.experiments.results import ExperimentRun
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _run_path(run_id: str):
    settings = get_settings()
    return settings.experiments_dir / f"{run_id}.json"


def save_run(run: ExperimentRun) -> str:
    """Write `run` to disk, returning the file path it was saved to."""
    path = _run_path(run.id)
    run.to_json_file(path)
    logger.info(f"Saved experiment run '{run.config.name}' ({len(run.items)} items) -> {path}")
    return str(path)


def load_run(run_id: str) -> ExperimentRun:
    """Load a run by id."""
    return ExperimentRun.from_json_file(_run_path(run_id))


def list_runs() -> list[ExperimentRun]:
    """Return all saved runs, most recently created first."""
    settings = get_settings()
    runs: list[ExperimentRun] = []
    for path in settings.experiments_dir.glob("*.json"):
        try:
            runs.append(ExperimentRun.from_json_file(path))
        except Exception as exc:
            logger.warning(f"Skipping unreadable experiment run file {path}: {exc}")
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs


def delete_run(run_id: str) -> bool:
    """Delete a run file by id. Returns True if a file was removed."""
    path = _run_path(run_id)
    if path.exists():
        path.unlink()
        logger.info(f"Deleted experiment run {run_id}")
        return True
    return False
