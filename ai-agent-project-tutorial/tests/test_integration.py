"""Integration test with small synthetic data (no real API calls).

Tests the end-to-end flow from chunking through retrieval, and verifies
that the Pipeline class can be instantiated and its components work together.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from rl_memory_retrieval.ingestion.chunker import chunk_text
from rl_memory_retrieval.models import MemoryItem, QueryItem
from rl_memory_retrieval.retrieval.retriever import baseline_retrieve
from rl_memory_retrieval.training.features import (
    _keyword_overlap,
    _entity_match,
    _topic_match,
    find_top_k_candidates,
    compute_state_dimension,
)


def test_end_to_end_chunking_to_retrieval():
    """Test the flow: chunk → simulate candidates → baseline retrieve."""
    text = "Astra robot has 18 hour battery. " * 20
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2

    # Simulate candidates with pre-computed features
    candidates = [
        {
            "sim": 0.9 - i * 0.1,
            "overlap": 0.2,
            "entity_match": 1.0 if i == 0 else 0.0,
            "topic_match": 0.0,
            "rank": i,
            "memory_id": i,
            "is_gold": 1.0 if i == 0 else 0.0,
            "text": chunks[i % len(chunks)],
        }
        for i in range(8)
    ]
    item = {
        "query": {
            "query_id": 0,
            "query": "What is battery?",
            "gold_memory_id": 0,
            "gold_value": "18h",
            "topic": "robotics",
            "entity": "Astra",
        },
        "candidates": candidates,
    }

    chosen = baseline_retrieve(item)
    assert chosen["memory_id"] == 0  # highest sim candidate


def test_feature_functions_integration():
    """Test that feature functions work correctly with real text."""
    query_text = "What is the battery life of Astra?"
    chunk_text_content = "Astra robot has 18 hour battery life."

    # Keyword overlap should find shared words
    overlap = _keyword_overlap(query_text, chunk_text_content)
    assert overlap > 0.0, "Should find overlapping keywords"

    # Entity match should find 'Astra'
    entity = _entity_match("Astra", chunk_text_content)
    assert entity == 1.0, "Astra should match"

    # Topic match
    topic = _topic_match("robotics", "robotics, automation")
    assert topic == 1.0, "Topic should match"

    # Missing entity/topic
    assert _entity_match("", chunk_text_content) == 0.0
    assert _topic_match("", "") == 0.0


def test_find_top_k_candidates_integration():
    """Test top-K candidate retrieval with synthetic embeddings."""
    dim = 8
    n_items = 10
    top_k = 5

    # Create items
    items = [MemoryItem(memory_id=i, text=f"memory {i}") for i in range(n_items)]

    # Create embeddings — item 3 is closest to query
    rng = np.random.default_rng(42)
    embeddings = rng.random((n_items, dim)).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    query_emb = embeddings[3].copy()  # Query is identical to item 3

    candidates, cand_embs, indices = find_top_k_candidates(
        query_emb, embeddings, items, top_k
    )

    assert len(candidates) == top_k
    assert candidates[0].memory_id == 3, "Closest match should be item 3"


def test_state_dimension():
    """Test state dimension computation."""
    assert compute_state_dimension(8) == 42  # 5*8 + 2
    assert compute_state_dimension(4) == 22  # 5*4 + 2
    assert compute_state_dimension(1) == 7   # 5*1 + 2


def test_pipeline_import():
    """Test that Pipeline can be imported from the package root."""
    from rl_memory_retrieval import Pipeline, Settings

    assert Pipeline is not None
    assert Settings is not None


def test_pipeline_instantiation():
    """Test Pipeline instantiation with default and custom settings."""
    from rl_memory_retrieval.pipeline import Pipeline
    from rl_memory_retrieval.config import Settings

    # Default config
    p1 = Pipeline()
    assert p1.config is not None
    assert p1.config.training.top_k_candidates == 8

    # Custom config
    settings = Settings(source="test.txt")
    p2 = Pipeline(config=settings)
    assert p2.config.source == "test.txt"


def test_cli_help():
    """Test that CLI --help works (smoke test)."""
    from click.testing import CliRunner
    from rl_memory_retrieval.cli import cli

    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "RL Memory Retrieval" in result.output

    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--source" in result.output

    result = runner.invoke(cli, ["query", "--help"])
    assert result.exit_code == 0
    assert "--question" in result.output
