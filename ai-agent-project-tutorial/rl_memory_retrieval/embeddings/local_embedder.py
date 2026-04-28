"""Local embedding backend using sentence-transformers."""

from __future__ import annotations

import numpy as np


class LocalEmbedder:
    """Local embedding backend using sentence-transformers.

    Runs entirely offline — no API key needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the local model.

        Args:
            texts: List of text strings to embed.

        Returns:
            L2-normalized numpy array of shape (len(texts), dimension).
        """
        arr = self._model.encode(texts, convert_to_numpy=True)
        arr = np.array(arr, dtype=np.float32)

        # L2 normalization
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms

        return arr

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension
