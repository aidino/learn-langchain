"""Retrievers for memory chunk selection.

Provides two retrieval strategies:
    - baseline_retrieve: Select the candidate with the highest cosine similarity.
    - rl_retrieve: Use a trained PPO model to select the best candidate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_memory_retrieval.training.features import (
    FEATURES_PER_CANDIDATE,
    compute_state_dimension,
)


def baseline_retrieve(item: dict) -> dict:
    """Select the candidate with the highest cosine similarity.

    This is the simplest retrieval strategy — it just picks the
    candidate with the highest pre-computed similarity score.

    Args:
        item: Dict containing "candidates" list. Each candidate has
              a "sim" key with the cosine similarity score.

    Returns:
        The candidate dict with the highest similarity.
    """
    candidates = item["candidates"]
    if not candidates:
        raise ValueError("No candidates available for retrieval")

    best_idx = max(range(len(candidates)), key=lambda i: candidates[i]["sim"])
    return candidates[best_idx]


def rl_retrieve(
    item: dict,
    model: Any,
    top_k: int = 8,
) -> dict:
    """Use a trained PPO model to select the best candidate.

    Builds the state feature vector from the pre-computed candidate
    features and uses model.predict(deterministic=True) to pick.

    Args:
        item: Dict containing "query" and "candidates".
        model: Trained SB3 PPO model with a predict() method.
        top_k: Number of candidate slots in the environment.

    Returns:
        The selected candidate dict.
    """
    candidates = item["candidates"][:top_k]
    query = item["query"]

    # Build state vector (same logic as environment._build_state)
    state_dim = compute_state_dimension(top_k)
    state = np.zeros(state_dim, dtype=np.float32)

    for i, cand in enumerate(candidates):
        offset = i * FEATURES_PER_CANDIDATE
        state[offset + 0] = max(0.0, min(1.0, cand.get("sim", 0.0)))
        state[offset + 1] = max(0.0, min(1.0, cand.get("overlap", 0.0)))
        state[offset + 2] = cand.get("entity_match", 0.0)
        state[offset + 3] = cand.get("topic_match", 0.0)
        state[offset + 4] = 1.0 / (1.0 + cand.get("rank", i))

    # Global features
    global_offset = FEATURES_PER_CANDIDATE * top_k
    query_text = query.get("query", "")
    tokens = query_text.split()
    state[global_offset + 0] = min(len(tokens) / 50.0, 1.0)
    state[global_offset + 1] = 1.0 if query.get("topic") else 0.0

    # Predict using the model
    action, _ = model.predict(state, deterministic=True)
    action_idx = int(action.item()) if hasattr(action, "item") else int(action)

    # Clamp to valid candidate range
    action_idx = min(action_idx, len(candidates) - 1)
    action_idx = max(action_idx, 0)

    return candidates[action_idx]
