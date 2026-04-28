"""Tests for the text chunker."""

from rl_memory_retrieval.ingestion.chunker import chunk_text


def test_chunk_basic():
    """Text longer than chunk_size should produce multiple chunks."""
    text = "A" * 600
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_overlap():
    """Consecutive chunks should share overlapping content."""
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    # Check overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        tail = chunks[i][-20:]
        assert tail in chunks[i + 1] or chunks[i + 1][:20] in chunks[i]


def test_chunk_short_text():
    """Short text should produce a single chunk."""
    chunks = chunk_text("short", chunk_size=300, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "short"


def test_chunk_empty_text():
    """Empty text should produce no chunks."""
    chunks = chunk_text("", chunk_size=300, chunk_overlap=50)
    assert len(chunks) == 0


def test_chunk_with_paragraphs():
    """Text with paragraph breaks should split on paragraph boundaries."""
    text = "Paragraph one content.\n\nParagraph two content.\n\nParagraph three content."
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) >= 2
