"""
Loaders that turn an uploaded file (PDF, CSV, or JSON) into a
normalized in-memory form the rest of the Dataset Builder can work
with:

- PDF  -> raw extracted text (one string per page, joined)
- CSV  -> list of row dicts (expects columns the user maps to
          question/ground_truth/context, or is treated as raw text)
- JSON -> list of record dicts, OR parsed directly as `EvalItem`s if
          the file already matches that shape

Each loader raises `DatasetLoadError` with a user-friendly message on
failure rather than leaking a raw parser traceback into the UI.
"""

from __future__ import annotations

import csv
import io
import json

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetLoadError(Exception):
    """Raised when an uploaded file can't be parsed into usable content."""


def load_pdf_text(file_bytes: bytes, filename: str = "document.pdf") -> str:
    """Extract plain text from a PDF's bytes, page by page."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DatasetLoadError(
            "pypdf is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)
            else:
                logger.debug(f"{filename}: page {page_number} had no extractable text")

        if not pages_text:
            raise DatasetLoadError(
                f"No extractable text found in '{filename}'. It may be a scanned/image-only PDF."
            )
        return "\n\n".join(pages_text)
    except DatasetLoadError:
        raise
    except Exception as exc:
        logger.error(f"Failed to parse PDF '{filename}': {exc}")
        raise DatasetLoadError(f"Could not read '{filename}' as a PDF: {exc}") from exc


def load_csv_records(file_bytes: bytes, filename: str = "data.csv") -> list[dict[str, str]]:
    """Parse CSV bytes into a list of row dicts, keyed by header."""
    try:
        text = file_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = [row for row in reader]
        if not rows:
            raise DatasetLoadError(f"'{filename}' has no data rows.")
        return rows
    except DatasetLoadError:
        raise
    except Exception as exc:
        logger.error(f"Failed to parse CSV '{filename}': {exc}")
        raise DatasetLoadError(f"Could not read '{filename}' as CSV: {exc}") from exc


def load_json_records(file_bytes: bytes, filename: str = "data.json") -> list[dict]:
    """Parse JSON bytes into a list of record dicts.

    Accepts either a top-level JSON array of objects, or a single
    object (wrapped into a one-item list).
    """
    try:
        parsed = json.loads(file_bytes.decode("utf-8"))
    except Exception as exc:
        logger.error(f"Failed to parse JSON '{filename}': {exc}")
        raise DatasetLoadError(f"Could not parse '{filename}' as JSON: {exc}") from exc

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        if not all(isinstance(item, dict) for item in parsed):
            raise DatasetLoadError(f"'{filename}' must be a JSON array of objects.")
        return parsed

    raise DatasetLoadError(f"'{filename}' must be a JSON object or array of objects.")


CSV_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("question", "query", "prompt"),
    "ground_truth": ("ground_truth", "answer", "expected_answer", "gold_answer"),
    "expected_context": ("expected_context", "context", "source_text", "passage"),
}


def map_record_to_fields(record: dict) -> dict[str, str]:
    """Best-effort map an arbitrary CSV/JSON row onto known EvalItem field names.

    Falls back to empty strings for fields it can't confidently find,
    leaving the rest to manual editing in the UI.
    """
    lowered = {str(k).strip().lower(): v for k, v in record.items()}
    mapped: dict[str, str] = {}
    for field, aliases in CSV_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in lowered and lowered[alias]:
                mapped[field] = str(lowered[alias])
                break
        else:
            mapped[field] = ""
    return mapped
