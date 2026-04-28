"""Tests for RL state feature engineering.

Uses synthetic data — no API calls needed.
"""

import numpy as np

from rl_memory_retrieval.models import MemoryItem, QueryItem
from rl_memory_retrieval.training.features import (
    FEATURES_PER_CANDIDATE,
    GLOBAL_FEATURES,
    compute_state,
    compute_state_dimension,
    find_top_k_candidates,
    _keyword_overlap,
    _entity_match,
    _topic_match,
)


def _make_item(memory_id: int, text: str, topic: str = "", entity: str = ""):
    return MemoryItem(memory_id=memory_id, text=text, topic=topic, entity=entity)


def _make_query(query_id: int, query: str, gold_id: int, topic: str = "", entity: str = ""):
    return QueryItem(
        query_id=query_id,
        query=query,
        gold_memory_id=gold_id,
        gold_value="answer",
        topic=topic,
        entity=entity,
    )


class TestStateDimension:
    def test_dimension_default(self):
        assert compute_state_dimension(8) == 42

    def test_dimension_small(self):
        assert compute_state_dimension(3) == 17

    def test_dimension_formula(self):
        for k in [1, 4, 8, 16]:
            assert compute_state_dimension(k) == FEATURES_PER_CANDIDATE * k + GLOBAL_FEATURES


class TestComputeState:
    def test_state_shape(self):
        """State vector should have correct shape."""
        top_k = 4
        query = _make_query(0, "what is the battery life", 0)
        q_emb = np.random.randn(16).astype(np.float32)
        candidates = [_make_item(i, f"chunk {i}") for i in range(top_k)]
        c_embs = np.random.randn(top_k, 16).astype(np.float32)

        state = compute_state(query, q_emb, candidates, c_embs, top_k)
        assert state.shape == (compute_state_dimension(top_k),)

    def test_state_padding(self):
        """Fewer candidates than top_k should pad with zeros."""
        top_k = 4
        query = _make_query(0, "test query", 0)
        q_emb = np.random.randn(16).astype(np.float32)
        candidates = [_make_item(0, "only one chunk")]
        c_embs = np.random.randn(1, 16).astype(np.float32)

        state = compute_state(query, q_emb, candidates, c_embs, top_k)

        # First candidate should have non-zero features
        assert state[4] > 0  # rank_score for first candidate
        # Second candidate slot should be zeros
        assert state[FEATURES_PER_CANDIDATE] == 0.0

    def test_rank_score_decreasing(self):
        """Rank score should decrease with position."""
        top_k = 3
        query = _make_query(0, "test", 0)
        q_emb = np.random.randn(16).astype(np.float32)
        candidates = [_make_item(i, f"chunk {i}") for i in range(3)]
        c_embs = np.random.randn(3, 16).astype(np.float32)

        state = compute_state(query, q_emb, candidates, c_embs, top_k)

        rank_scores = [state[i * FEATURES_PER_CANDIDATE + 4] for i in range(3)]
        assert rank_scores[0] > rank_scores[1] > rank_scores[2]

    def test_global_features(self):
        """Global features should be at the end of the state vector."""
        top_k = 2
        query = _make_query(0, "a b c d e", 0, topic="batteries")
        q_emb = np.random.randn(16).astype(np.float32)
        candidates = [_make_item(0, "chunk")]
        c_embs = np.random.randn(1, 16).astype(np.float32)

        state = compute_state(query, q_emb, candidates, c_embs, top_k)

        global_offset = FEATURES_PER_CANDIDATE * top_k
        # query_length = 5 / 50 = 0.1
        assert abs(state[global_offset] - 0.1) < 1e-6
        # topic_present = 1.0
        assert state[global_offset + 1] == 1.0

    def test_no_topic_global(self):
        """topic_present should be 0.0 when no topic."""
        top_k = 2
        query = _make_query(0, "test", 0, topic="")
        q_emb = np.random.randn(16).astype(np.float32)
        state = compute_state(query, q_emb, [], np.empty((0, 16)), top_k)

        global_offset = FEATURES_PER_CANDIDATE * top_k
        assert state[global_offset + 1] == 0.0


class TestKeywordOverlap:
    def test_identical(self):
        assert _keyword_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _keyword_overlap("cat dog", "fish bird") == 0.0

    def test_partial_overlap(self):
        result = _keyword_overlap("the quick brown", "the slow brown fox")
        assert 0.0 < result < 1.0

    def test_empty_strings(self):
        assert _keyword_overlap("", "hello") == 0.0
        assert _keyword_overlap("hello", "") == 0.0


class TestEntityMatch:
    def test_entity_found(self):
        assert _entity_match("Pulse", "The Pulse device has a long battery life") == 1.0

    def test_entity_not_found(self):
        assert _entity_match("Pulse", "The device has a long battery life") == 0.0

    def test_empty_entity(self):
        assert _entity_match("", "some text") == 0.0

    def test_multiple_entities(self):
        assert _entity_match("Apple, Samsung", "Samsung Galaxy is great") == 1.0

    def test_case_insensitive(self):
        assert _entity_match("PULSE", "pulse smartwatch features") == 1.0


class TestTopicMatch:
    def test_matching_topic(self):
        assert _topic_match("batteries", "batteries") == 1.0

    def test_no_match(self):
        assert _topic_match("batteries", "display") == 0.0

    def test_empty_topics(self):
        assert _topic_match("", "batteries") == 0.0
        assert _topic_match("batteries", "") == 0.0

    def test_multiple_topics_overlap(self):
        assert _topic_match("batteries, display", "display, camera") == 1.0

    def test_multiple_topics_no_overlap(self):
        assert _topic_match("batteries", "display, camera") == 0.0


class TestFindTopK:
    def test_returns_correct_count(self):
        items = [_make_item(i, f"item {i}") for i in range(10)]
        embs = np.random.randn(10, 16).astype(np.float32)
        q_emb = np.random.randn(16).astype(np.float32)

        candidates, cand_embs, indices = find_top_k_candidates(q_emb, embs, items, 3)
        assert len(candidates) == 3
        assert cand_embs.shape == (3, 16)
        assert len(indices) == 3

    def test_most_similar_first(self):
        """First candidate should be the most similar."""
        items = [_make_item(i, f"item {i}") for i in range(5)]
        embs = np.eye(5, dtype=np.float32)[:, :5]  # 5 orthogonal vectors
        q_emb = embs[2]  # Same as item 2

        candidates, _, indices = find_top_k_candidates(q_emb, embs, items, 3)
        assert candidates[0].memory_id == 2

    def test_top_k_exceeds_items(self):
        """Should handle top_k larger than number of items."""
        items = [_make_item(i, f"item {i}") for i in range(3)]
        embs = np.random.randn(3, 16).astype(np.float32)
        q_emb = np.random.randn(16).astype(np.float32)

        candidates, cand_embs, indices = find_top_k_candidates(q_emb, embs, items, 5)
        assert len(candidates) == 3  # Can't return more than available
