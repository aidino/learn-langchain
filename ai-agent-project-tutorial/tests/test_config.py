"""Tests for the configuration system."""

from rl_memory_retrieval.config import Settings


def test_default_settings():
    """Verify defaults load correctly without a config file."""
    s = Settings(source="https://example.com")
    assert s.embeddings.backend == "openai"
    assert s.training.top_k_candidates == 8
    assert s.chunking.chunk_size == 300


def test_override_settings():
    """Verify init kwargs override defaults."""
    from rl_memory_retrieval.config import ChunkingConfig

    s = Settings(
        source="https://example.com",
        chunking=ChunkingConfig(chunk_size=500, chunk_overlap=100),
    )
    assert s.chunking.chunk_size == 500
    assert s.chunking.chunk_overlap == 100


def test_all_nested_defaults():
    """Verify all nested config models have expected defaults."""
    s = Settings(source="test")
    assert s.embeddings.model == "text-embedding-3-small"
    assert s.extraction.enabled is False
    assert s.query_generation.queries_per_chunk == 3
    assert s.training.learning_rate == 3e-4
    assert s.training.total_timesteps == 12000
    assert s.evaluation.sample_size == 20
    assert s.evaluation.use_llm_judge is True
