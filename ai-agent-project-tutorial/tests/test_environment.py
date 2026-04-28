"""Tests for the Gymnasium RL environment.

Uses synthetic data — no API calls needed.
"""

import numpy as np
import gymnasium as gym

from rl_memory_retrieval.training.environment import MemoryRetrievalEnv
from rl_memory_retrieval.training.features import compute_state_dimension


def _make_dummy_items(n: int = 5, top_k: int = 8) -> list[dict]:
    """Create dummy training items for the environment.

    Each item has a query dict and a list of candidate dicts.
    Candidate 0 is always the gold (correct) candidate.
    """
    items = []
    for i in range(n):
        candidates = []
        for j in range(top_k):
            candidates.append({
                "sim": 0.9 - 0.05 * j,
                "overlap": 0.2 if j == 0 else 0.1,
                "entity_match": 1.0 if j == 0 else 0.0,
                "topic_match": 1.0 if j == 0 else 0.0,
                "rank": j,
                "memory_id": j,
                "is_gold": 1.0 if j == 0 else 0.0,
                "text": f"memory chunk {j} for query {i}",
            })
        items.append({
            "query": {
                "query_id": i,
                "query": f"test question {i}?",
                "gold_memory_id": 0,
                "gold_value": f"answer {i}",
                "topic": "test_topic",
                "entity": "test_entity",
            },
            "candidates": candidates,
        })
    return items


class TestEnvironmentInit:
    def test_creates_valid_gym_env(self):
        """Environment should be a valid Gymnasium env."""
        env = MemoryRetrievalEnv(_make_dummy_items())
        assert isinstance(env, gym.Env)

    def test_observation_space(self):
        """Observation space should match state dimension."""
        top_k = 8
        env = MemoryRetrievalEnv(_make_dummy_items(top_k=top_k), top_k=top_k)
        expected_dim = compute_state_dimension(top_k)
        assert env.observation_space.shape == (expected_dim,)

    def test_action_space(self):
        """Action space should be Discrete(top_k)."""
        top_k = 8
        env = MemoryRetrievalEnv(_make_dummy_items(top_k=top_k), top_k=top_k)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == top_k


class TestEnvironmentReset:
    def test_reset_returns_obs_and_info(self):
        env = MemoryRetrievalEnv(_make_dummy_items())
        obs, info = env.reset()
        assert obs.shape == (compute_state_dimension(8),)
        assert obs.dtype == np.float32
        assert "query_id" in info

    def test_reset_with_seed(self):
        env = MemoryRetrievalEnv(_make_dummy_items())
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestEnvironmentStep:
    def test_step_correct_action(self):
        """Selecting the gold candidate should give positive reward."""
        env = MemoryRetrievalEnv(_make_dummy_items())
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(0)  # 0 is gold
        assert terminated is True
        assert truncated is False
        assert reward > 0
        assert info["is_correct"] is True

    def test_step_wrong_action(self):
        """Selecting a non-gold candidate should give lower reward."""
        env = MemoryRetrievalEnv(_make_dummy_items())
        env.reset(seed=0)
        _, reward, terminated, _, info = env.step(7)  # 7 is not gold
        assert terminated is True
        assert info["is_correct"] is False

    def test_correct_reward_higher_than_wrong(self):
        """Correct action should give higher reward than wrong action."""
        items = _make_dummy_items()
        env = MemoryRetrievalEnv(items)

        env.reset(seed=0)
        _, reward_correct, _, _, _ = env.step(0)

        env.reset(seed=0)
        _, reward_wrong, _, _, _ = env.step(7)

        assert reward_correct > reward_wrong

    def test_step_returns_new_observation(self):
        """After step, observation should still have correct shape."""
        env = MemoryRetrievalEnv(_make_dummy_items())
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (compute_state_dimension(8),)


class TestEnvironmentEpisode:
    def test_one_step_episode(self):
        """Each episode should be exactly one step (single action)."""
        env = MemoryRetrievalEnv(_make_dummy_items())
        env.reset()
        _, _, terminated, _, _ = env.step(0)
        assert terminated is True

    def test_full_cycle(self):
        """Can reset and step through multiple episodes."""
        items = _make_dummy_items(n=3)
        env = MemoryRetrievalEnv(items)

        for _ in range(10):  # More episodes than items to test cycling
            obs, info = env.reset()
            assert obs.shape[0] > 0
            _, _, terminated, _, _ = env.step(0)
            assert terminated is True

    def test_reward_components_in_range(self):
        """Reward should be bounded and reasonable."""
        env = MemoryRetrievalEnv(_make_dummy_items())
        env.reset()
        _, reward, _, _, _ = env.step(0)
        # Max possible reward: 2.0 + 0.8 + 0.6 + 0.5 + 0.3 - 0 = 4.2
        assert -1.0 <= reward <= 5.0
