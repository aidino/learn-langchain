"""LLM-based QA pair generation from knowledge chunks.

Generates diverse question-answer pairs from each MemoryItem using an LLM.
Includes deduplication based on cosine similarity of query embeddings.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from rl_memory_retrieval.models import MemoryItem, QueryItem

if TYPE_CHECKING:
    from rl_memory_retrieval.embeddings.base import Embedder

_SYSTEM_PROMPT = """Generate {n} diverse question-answer pairs from this text.
Each question should require information from the text to answer.
Questions should be varied in style (factual, analytical, what/how/why).
Return ONLY a JSON array of objects with "question" and "answer" keys.

Example format:
[
  {{"question": "What is the main topic?", "answer": "The main topic is..."}},
  {{"question": "How does X work?", "answer": "X works by..."}}
]"""


def generate_queries(
    items: list[MemoryItem],
    queries_per_chunk: int = 3,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    embedder: Embedder | None = None,
    dedup_threshold: float = 0.9,
) -> list[QueryItem]:
    """Generate QA pairs from MemoryItems using an LLM.

    Args:
        items: The memory chunks to generate queries from.
        queries_per_chunk: Number of QA pairs to request per chunk.
        model: The LLM model to use for generation.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        embedder: Optional embedder for query deduplication.
        dedup_threshold: Cosine similarity threshold for deduplication.

    Returns:
        A list of QueryItem objects with gold references.
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=key)

    all_queries: list[QueryItem] = []
    query_id_counter = 0

    for item in tqdm(items, desc="Generating queries"):
        system = _SYSTEM_PROMPT.format(n=queries_per_chunk)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": item.text},
                ],
                temperature=0.7,
            )

            content = response.choices[0].message.content or ""
            pairs = parse_qa_response(content)

            for pair in pairs[:queries_per_chunk]:
                qi = QueryItem(
                    query_id=query_id_counter,
                    query=pair["question"],
                    gold_memory_id=item.memory_id,
                    gold_value=pair["answer"],
                    topic=item.topic,
                    entity=item.entity,
                )
                all_queries.append(qi)
                query_id_counter += 1

        except Exception:
            # Skip chunks that fail generation — non-fatal
            continue

    # Deduplication step
    if embedder is not None and len(all_queries) > 1:
        all_queries = _deduplicate(all_queries, embedder, dedup_threshold)

    return all_queries


def parse_qa_response(content: str) -> list[dict[str, str]]:
    """Parse a JSON array of QA pairs from LLM response.

    Handles common LLM output quirks: markdown code fences, trailing commas,
    and partial JSON.

    Args:
        content: Raw LLM response text.

    Returns:
        A list of dicts with 'question' and 'answer' keys.
    """
    # Strip markdown code fences if present
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(parsed, list):
        return []

    # Validate each item has the required keys
    result = []
    for item in parsed:
        if isinstance(item, dict) and "question" in item and "answer" in item:
            result.append(
                {"question": str(item["question"]), "answer": str(item["answer"])}
            )

    return result


def _deduplicate(
    queries: list[QueryItem],
    embedder: Embedder,
    threshold: float,
) -> list[QueryItem]:
    """Remove near-duplicate queries based on embedding similarity.

    Args:
        queries: List of QueryItems to deduplicate.
        embedder: Embedder to compute query embeddings.
        threshold: Cosine similarity threshold (queries above this are dupes).

    Returns:
        Deduplicated list of QueryItems.
    """
    texts = [q.query for q in queries]
    embeddings = embedder.embed(texts)

    # Compute pairwise cosine similarities
    # Embeddings are already L2-normalized, so dot product = cosine similarity
    keep_mask = [True] * len(queries)

    for i in range(len(queries)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(queries)):
            if not keep_mask[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim > threshold:
                keep_mask[j] = False

    return [q for q, keep in zip(queries, keep_mask) if keep]
