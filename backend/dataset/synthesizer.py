"""
Synthetic evaluation-dataset generation.

Given raw source text, this module chunks it and asks the configured
LLM to produce question/ground-truth/expected-context triples per
chunk — the seed data an evaluation dataset needs before any
retrieval or generation has happened.

The LLM is prompted to return strict JSON so parsing is deterministic;
`_parse_llm_json` is defensive about the model wrapping its answer in
markdown fences, which happens often enough in practice to guard against.
"""

from __future__ import annotations

import json
import re

from backend.dataset.chunking import chunk_text
from backend.dataset.schema import EvalItem
from backend.utils.llm_client import LLMClient, LLMError
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert dataset curator building evaluation data for a "
    "Retrieval-Augmented Generation (RAG) system. Given a passage of text, "
    "generate realistic questions a user might ask that this passage answers, "
    "along with a correct, concise ground-truth answer grounded ONLY in the "
    "passage. Never invent facts not present in the passage."
)

_USER_PROMPT_TEMPLATE = """\
Passage:
\"\"\"
{chunk}
\"\"\"

Generate exactly {n} question/answer pair(s) grounded in this passage.

Respond with ONLY a JSON array (no markdown fences, no commentary), where each \
element has this exact shape:
{{"question": "...", "ground_truth": "...", "expected_chunk": "the exact sentence(s) from the passage that answer the question"}}
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_llm_json(raw_text: str) -> list[dict]:
    """Extract a JSON array from an LLM response, tolerating markdown fences."""
    candidate = raw_text.strip()
    fence_match = _JSON_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc

    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array of question/answer objects.")
    return parsed


def generate_synthetic_items(
    source_text: str,
    source_document: str,
    client: LLMClient,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    questions_per_chunk: int = 2,
    max_chunks: int = 20,
) -> tuple[list[EvalItem], list[str]]:
    """Generate synthetic `EvalItem`s from `source_text`.

    Returns `(items, warnings)` — generation continues past a single
    chunk's failure (e.g. a malformed LLM response) rather than
    aborting the whole document; each failure is recorded as a warning
    string for display in the UI.
    """
    chunks = chunk_text(source_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return [], [f"No usable text found in '{source_document}'."]

    if len(chunks) > max_chunks:
        logger.warning(f"{source_document}: {len(chunks)} chunks exceeds max_chunks={max_chunks}, truncating")
        chunks = chunks[:max_chunks]

    items: list[EvalItem] = []
    warnings: list[str] = []

    for chunk in chunks:
        prompt = _USER_PROMPT_TEMPLATE.format(chunk=chunk.text, n=questions_per_chunk)
        try:
            response = client.complete(prompt=prompt, system=_SYSTEM_PROMPT, temperature=0.4)
            records = _parse_llm_json(response.text)
        except LLMError as exc:
            warnings.append(f"Chunk {chunk.chunk_index}: LLM call failed — {exc}")
            continue
        except ValueError as exc:
            warnings.append(f"Chunk {chunk.chunk_index}: could not parse response — {exc}")
            continue

        for record in records:
            question = str(record.get("question", "")).strip()
            ground_truth = str(record.get("ground_truth", "")).strip()
            expected_chunk = str(record.get("expected_chunk", "")).strip() or chunk.text

            if not question or not ground_truth:
                warnings.append(f"Chunk {chunk.chunk_index}: skipped a malformed item (missing question/answer)")
                continue

            items.append(
                EvalItem(
                    question=question,
                    ground_truth=ground_truth,
                    expected_context=chunk.text,
                    expected_chunk=expected_chunk,
                    source_document=source_document,
                    source="synthetic",
                )
            )

    return items, warnings
