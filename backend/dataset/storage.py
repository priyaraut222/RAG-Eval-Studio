"""
Persistence for `EvalDataset`s.

Datasets are stored as one JSON file per dataset under
`data/datasets/<id>.json`. This module is the only place that knows
that detail — everything else (the Streamlit page, future experiment
runner) goes through these functions.
"""

from __future__ import annotations

from backend.config.settings import get_settings
from backend.dataset.schema import EvalDataset
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _dataset_path(dataset_id: str):
    settings = get_settings()
    return settings.datasets_dir / f"{dataset_id}.json"


def save_dataset(dataset: EvalDataset) -> str:
    """Write `dataset` to disk, returning the file path it was saved to."""
    path = _dataset_path(dataset.id)
    dataset.to_json_file(path)
    logger.info(f"Saved dataset '{dataset.name}' ({dataset.size} items) -> {path}")
    return str(path)


def load_dataset(dataset_id: str) -> EvalDataset:
    """Load a dataset by id."""
    return EvalDataset.from_json_file(_dataset_path(dataset_id))


def list_datasets() -> list[EvalDataset]:
    """Return all saved datasets, most recently created first."""
    settings = get_settings()
    datasets: list[EvalDataset] = []
    for path in settings.datasets_dir.glob("*.json"):
        try:
            datasets.append(EvalDataset.from_json_file(path))
        except Exception as exc:
            logger.warning(f"Skipping unreadable dataset file {path}: {exc}")
    datasets.sort(key=lambda d: d.created_at, reverse=True)
    return datasets


def delete_dataset(dataset_id: str) -> bool:
    """Delete a dataset file by id. Returns True if a file was removed."""
    path = _dataset_path(dataset_id)
    if path.exists():
        path.unlink()
        logger.info(f"Deleted dataset {dataset_id}")
        return True
    return False
