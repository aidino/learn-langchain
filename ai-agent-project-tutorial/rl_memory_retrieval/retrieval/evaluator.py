"""Evaluation module for comparing retrieval strategies.

Computes accuracy and optional LLM-judged QA evaluation.
"""

from __future__ import annotations

from typing import Callable


def evaluate_retriever(
    items: list[dict],
    retriever_fn: Callable[[dict], dict],
) -> dict:
    """Evaluate a retriever function on a set of items.

    Args:
        items: List of items, each with "query" and "candidates".
        retriever_fn: A function that takes an item and returns the
                      chosen candidate dict.

    Returns:
        Dict with evaluation metrics:
            - accuracy: fraction of correct retrievals
            - correct: count of correct retrievals
            - total: total number of items evaluated
            - details: list of per-item results
    """
    correct = 0
    total = len(items)
    details = []

    for item in items:
        chosen = retriever_fn(item)

        is_correct = chosen.get("is_gold", 0.0) == 1.0
        if is_correct:
            correct += 1

        details.append({
            "query_id": item["query"]["query_id"],
            "chosen_memory_id": chosen["memory_id"],
            "gold_memory_id": item["query"]["gold_memory_id"],
            "is_correct": is_correct,
            "sim": chosen.get("sim", 0.0),
        })

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


def compare_retrievers(
    items: list[dict],
    retrievers: dict[str, Callable[[dict], dict]],
) -> dict[str, dict]:
    """Compare multiple retrieval strategies.

    Args:
        items: List of evaluation items.
        retrievers: Dict mapping retriever name to retriever function.

    Returns:
        Dict mapping retriever name to evaluation results.
    """
    results = {}
    for name, fn in retrievers.items():
        results[name] = evaluate_retriever(items, fn)
    return results
