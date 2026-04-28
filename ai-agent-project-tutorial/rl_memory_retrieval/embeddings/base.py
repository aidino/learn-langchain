"""Embedder protocol defining the interface for all embedding backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Protocol for embedding backends.

    All embedders must implement `embed()` and the `dimension` property.
    """

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a numpy array.

        Args:
            texts: List of text strings to embed.

        Returns:
            A 2D numpy array of shape (len(texts), dimension), L2-normalized.
        """
        ...

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...
