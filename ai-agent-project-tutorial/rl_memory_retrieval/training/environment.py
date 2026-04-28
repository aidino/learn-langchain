"""Gymnasium RL environment for memory retrieval.

One-step episodes: the agent sees feature representations of the top-K
candidate chunks for a query and must pick the best one.

Reward function from spec:
    reward = 2.0 * is_gold
           + 0.8 * entity_match
           + 0.6 * topic_match
           + 0.5 * cosine_sim
           + 0.3 * keyword_overlap
           - 0.15 * rank
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_memory_retrieval.training.features import (
    FEATURES_PER_CANDIDATE,
    GLOBAL_FEATURES,
    compute_state_dimension,
)


class MemoryRetrievalEnv(gym.Env):
    """Gymnasium environment for RL-based memory retrieval.

    Each episode presents one query with its top-K candidate chunks.
    The agent takes a single action (selecting one candidate) and
    receives a reward based on how good the selection is.

    Args:
        items: List of training items, each containing:
            - "query": dict with query_id, query, gold_memory_id,
              gold_value, topic, entity
            - "candidates": list of dicts, each with sim, overlap,
              entity_match, topic_match, rank, memory_id, is_gold, text
        top_k: Number of candidate slots. Defaults to 8.
        seed: Random seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        items: list[dict],
        top_k: int = 8,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self._items = items
        self._top_k = top_k
        self._state_dim = compute_state_dimension(top_k)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self._state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(top_k)

        # Episode state
        self._current_idx: int = 0
        self._current_state: np.ndarray | None = None
        self._current_candidates: list[dict] = []
        self._current_query: dict = {}
        self._rng = np.random.default_rng(seed)
        self._order: np.ndarray = np.arange(len(items))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a new episode (next query).

        Args:
            seed: Optional seed for reproducibility.
            options: Unused, present for Gymnasium API compliance.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._order = self._rng.permutation(len(self._items))
            self._current_idx = 0

        # Wrap around when all items have been seen
        if self._current_idx >= len(self._items):
            self._order = self._rng.permutation(len(self._items))
            self._current_idx = 0

        item_idx = int(self._order[self._current_idx])
        item = self._items[item_idx]

        self._current_query = item["query"]
        self._current_candidates = item["candidates"][:self._top_k]

        # Build the state feature vector
        self._current_state = self._build_state(
            self._current_candidates, self._current_query
        )

        info = {
            "query_id": self._current_query["query_id"],
            "query": self._current_query["query"],
        }

        return self._current_state.copy(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take an action (select a candidate) and compute reward.

        Args:
            action: Index of the chosen candidate (0 to top_k-1).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self._current_state is not None, "Must call reset() before step()"
        assert 0 <= action < self._top_k, f"Action {action} out of range [0, {self._top_k})"

        # Get the chosen candidate
        if action < len(self._current_candidates):
            chosen = self._current_candidates[action]
        else:
            # Action points to a padded (empty) slot — bad choice
            chosen = {
                "sim": 0.0,
                "overlap": 0.0,
                "entity_match": 0.0,
                "topic_match": 0.0,
                "rank": action,
                "memory_id": -1,
                "is_gold": 0.0,
                "text": "",
            }

        # Compute reward
        reward = self._compute_reward(chosen)

        # Check correctness
        is_correct = chosen.get("is_gold", 0.0) == 1.0

        # One-step episode — always terminates
        terminated = True
        truncated = False

        # Advance to next item for the next episode
        self._current_idx += 1

        info = {
            "is_correct": is_correct,
            "chosen_memory_id": chosen["memory_id"],
            "gold_memory_id": self._current_query["gold_memory_id"],
            "reward_breakdown": {
                "is_gold": 2.0 * chosen.get("is_gold", 0.0),
                "entity_match": 0.8 * chosen.get("entity_match", 0.0),
                "topic_match": 0.6 * chosen.get("topic_match", 0.0),
                "cosine_sim": 0.5 * chosen.get("sim", 0.0),
                "keyword_overlap": 0.3 * chosen.get("overlap", 0.0),
                "rank_penalty": -0.15 * chosen.get("rank", 0),
            },
        }

        return self._current_state.copy(), reward, terminated, truncated, info

    def _compute_reward(self, candidate: dict) -> float:
        """Compute the reward for selecting a candidate.

        Reward function from spec:
            reward = 2.0 * is_gold
                   + 0.8 * entity_match
                   + 0.6 * topic_match
                   + 0.5 * cosine_sim
                   + 0.3 * keyword_overlap
                   - 0.15 * rank

        Args:
            candidate: Dict with candidate features.

        Returns:
            Scalar reward value.
        """
        reward = (
            2.0 * candidate.get("is_gold", 0.0)
            + 0.8 * candidate.get("entity_match", 0.0)
            + 0.6 * candidate.get("topic_match", 0.0)
            + 0.5 * candidate.get("sim", 0.0)
            + 0.3 * candidate.get("overlap", 0.0)
            - 0.15 * candidate.get("rank", 0)
        )
        return float(reward)

    def _build_state(
        self, candidates: list[dict], query: dict
    ) -> np.ndarray:
        """Build the state feature vector from pre-computed candidate features.

        This method uses the pre-computed features stored in the candidate
        dicts (sim, overlap, entity_match, topic_match, rank) rather than
        recomputing from raw embeddings. The features module's compute_state
        is used during dataset preparation; here we just assemble the vector.

        Args:
            candidates: List of candidate feature dicts.
            query: Query dict with query text, topic, entity.

        Returns:
            1D state vector of shape (state_dim,).
        """
        state = np.zeros(self._state_dim, dtype=np.float32)

        for i, cand in enumerate(candidates[:self._top_k]):
            offset = i * FEATURES_PER_CANDIDATE

            # Per-candidate features
            state[offset + 0] = max(0.0, min(1.0, cand.get("sim", 0.0)))
            state[offset + 1] = max(0.0, min(1.0, cand.get("overlap", 0.0)))
            state[offset + 2] = cand.get("entity_match", 0.0)
            state[offset + 3] = cand.get("topic_match", 0.0)
            state[offset + 4] = 1.0 / (1.0 + cand.get("rank", i))

        # Global features
        global_offset = FEATURES_PER_CANDIDATE * self._top_k
        query_text = query.get("query", "")
        tokens = query_text.split()
        state[global_offset + 0] = min(len(tokens) / 50.0, 1.0)
        state[global_offset + 1] = 1.0 if query.get("topic") else 0.0

        return state
