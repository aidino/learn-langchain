"""OpenAI embedding backend."""

from __future__ import annotations

import numpy as np
from openai import OpenAI


# Known default dimensions for common models
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _chunked(xs: list, n: int):
    """Yield successive n-sized chunks from xs."""
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


class OpenAIEmbedder:
    """OpenAI API embedding backend.

    Uses the OpenAI embeddings API to convert texts to vectors,
    with L2 normalization applied post-embedding.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 64,
    ):
        self._model = model
        self._batch_size = batch_size
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._dimension = _MODEL_DIMS.get(model, 0)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the OpenAI API.

        Args:
            texts: List of text strings to embed.

        Returns:
            L2-normalized numpy array of shape (len(texts), dimension).
        """
        outputs: list[list[float]] = []
        for batch in _chunked(texts, self._batch_size):
            resp = self._client.embeddings.create(model=self._model, input=batch)
            batch_vecs = [d.embedding for d in resp.data]
            outputs.extend(batch_vecs)

        arr = np.array(outputs, dtype=np.float32)

        # L2 normalization
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms

        # Cache dimension from first call if not known
        if self._dimension == 0 and arr.shape[0] > 0:
            self._dimension = arr.shape[1]

        return arr

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension
