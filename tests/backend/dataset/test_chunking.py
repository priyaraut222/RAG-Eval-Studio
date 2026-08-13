from backend.dataset.chunking import chunk_text


def test_chunk_text_empty_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short_text_single_chunk():
    text = "A short sentence."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0


def test_chunk_text_respects_chunk_size_with_word_snap():
    text = "word " * 200  # 1000 chars
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 100


def test_chunk_text_overlap_shares_content_between_consecutive_chunks():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    chunks = chunk_text(text, chunk_size=80, chunk_overlap=30)
    assert len(chunks) >= 2
    # Consecutive chunks should overlap in start position, proving forward progress
    # without skipping content.
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.start_char < earlier.end_char


def test_chunk_text_indices_are_sequential():
    text = "sentence one. " * 50
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_text_rejects_non_positive_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0)


def test_chunk_text_clamps_overlap_to_avoid_infinite_loop():
    # overlap >= chunk_size should be clamped rather than looping forever
    chunks = chunk_text("word " * 500, chunk_size=50, chunk_overlap=999)
    assert len(chunks) > 0
