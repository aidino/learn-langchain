"""Embeddings layer with pluggable backends."""

from __future__ import annotations

from rl_memory_retrieval.embeddings.base import Embedder
from rl_memory_retrieval.config import EmbeddingsConfig


def create_embedder(config: EmbeddingsConfig) -> Embedder:
    """Create an embedder instance based on config.

    Args:
        config: Embeddings configuration specifying backend and model.

    Returns:
        An Embedder instance (OpenAI or local).

    Raises:
        ValueError: If the backend is not recognized.
    """
    if config.backend == "openai":
        from rl_memory_retrieval.embeddings.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(model=config.model, batch_size=config.batch_size)
    elif config.backend == "local":
        from rl_memory_retrieval.embeddings.local_embedder import LocalEmbedder

        return LocalEmbedder(model_name=config.model)
    else:
        raise ValueError(f"Unknown embeddings backend: {config.backend}")


__all__ = ["Embedder", "create_embedder"]
