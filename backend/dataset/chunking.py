"""
Text chunking utilities.

A small, dependency-free splitter used to break long documents into
overlapping chunks — both for synthetic dataset generation
(`expected_chunk`) and later, in Phase 3+, for building the actual
retrieval index. Kept intentionally simple (character-based with a
word-boundary snap) rather than pulling in a heavier splitter, so
Phase 1/2 have no hard dependency on LangChain's text splitters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """One chunk of a source document, with its position for traceability."""

    text: str
    start_char: int
    end_char: int
    chunk_index: int


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """Split `text` into overlapping chunks of roughly `chunk_size` characters.

    Splits are snapped to the nearest whitespace so words aren't cut
    mid-token where possible. `chunk_overlap` must be smaller than
    `chunk_size`; it's silently clamped otherwise to avoid infinite loops.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    text_length = len(text)
    index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Snap to the last whitespace before `end`, unless we're at the doc end.
        if end < text_length:
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap

        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, start_char=start, end_char=end, chunk_index=index))
            index += 1

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)  # guarantee forward progress

    return chunks
