"""Tests for retrieval and evaluation.

Uses synthetic data — no API calls needed.
"""

import numpy as np
from unittest.mock import MagicMock

from rl_memory_retrieval.retrieval.retriever import baseline_retrieve, rl_retrieve
from rl_memory_retrieval.retrieval.evaluator import evaluate_retriever


def _make_item(gold_idx: int = 0, n_candidates: int = 3) -> dict:
    """Create a single test item with candidates."""
    candidates = []
    for j in range(n_candidates):
        candidates.append({
            "sim": 0.3 + 0.3 * (n_candidates - j - 1) / max(n_candidates - 1, 1),
            "overlap": 0.1,
            "entity_match": 1.0 if j == gold_idx else 0.0,
            "topic_match": 0.0,
            "rank": j,
            "memory_id": j,
            "is_gold": 1.0 if j == gold_idx else 0.0,
            "text": f"memory chunk {j}",
        })
    return {
        "query": {
            "query_id": 0,
            "query": "test question?",
            "gold_memory_id": gold_idx,
            "gold_value": "answer",
            "topic": "t",
            "entity": "e",
        },
        "candidates": candidates,
    }


class TestBaselineRetrieve:
    def test_selects_highest_similarity(self):
        """Baseline should pick the candidate with highest sim."""
        item = _make_item()
        # Candidate 0 has highest sim (0.6), candidate 2 has lowest (0.3)
        chosen = baseline_retrieve(item)
        assert chosen["memory_id"] == 0  # highest sim

    def test_returns_candidate_dict(self):
        item = _make_item()
        chosen = baseline_retrieve(item)
        assert "memory_id" in chosen
        assert "sim" in chosen
        assert "text" in chosen

    def test_single_candidate(self):
        item = _make_item(n_candidates=1)
        chosen = baseline_retrieve(item)
        assert chosen["memory_id"] == 0


class TestRLRetrieve:
    def test_uses_model_prediction(self):
        """RL retrieve should use model.predict to select."""
        item = _make_item()

        # Mock a PPO model
        mock_model = MagicMock()
        # predict returns (action_array, state) — select candidate 1
        mock_model.predict.return_value = (np.array([1]), None)

        chosen = rl_retrieve(item, mock_model, top_k=3)
        assert chosen["memory_id"] == 1
        mock_model.predict.assert_called_once()

    def test_deterministic_prediction(self):
        """rl_retrieve should pass deterministic=True to model."""
        item = _make_item()
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0]), None)

        rl_retrieve(item, mock_model, top_k=3)

        _, kwargs = mock_model.predict.call_args
        assert kwargs.get("deterministic") is True


class TestEvaluateRetriever:
    def test_accuracy_calculation(self):
        """Evaluate should compute correct accuracy."""
        # All correct items
        items = [_make_item(gold_idx=0) for _ in range(5)]

        # Baseline always picks highest sim (index 0), which is the gold
        results = evaluate_retriever(items, retriever_fn=baseline_retrieve)

        assert results["accuracy"] == 1.0
        assert results["total"] == 5
        assert results["correct"] == 5

    def test_partial_accuracy(self):
        """Evaluate with mix of correct/incorrect."""
        # Item where gold is NOT the highest sim
        items = []
        for i in range(4):
            item = _make_item(gold_idx=0)
            items.append(item)

        # Override 2 items: gold at index 2 but highest sim at index 0
        for i in range(2):
            items[i]["candidates"][0]["is_gold"] = 0.0
            items[i]["candidates"][2]["is_gold"] = 1.0
            items[i]["query"]["gold_memory_id"] = 2

        results = evaluate_retriever(items, retriever_fn=baseline_retrieve)

        # Baseline picks index 0 (highest sim). Only 2 of 4 have gold at 0.
        assert results["accuracy"] == 0.5
        assert results["correct"] == 2

    def test_returns_required_keys(self):
        items = [_make_item() for _ in range(3)]
        results = evaluate_retriever(items, retriever_fn=baseline_retrieve)

        assert "accuracy" in results
        assert "total" in results
        assert "correct" in results
