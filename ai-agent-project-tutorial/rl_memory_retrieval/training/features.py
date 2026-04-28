"""RL state feature engineering for the memory retrieval environment.

Computes feature vectors that represent the state for each query-candidate pair.
The state vector is composed of per-candidate features and global features.

Per candidate (5 features × top_k):
    - cosine_similarity: vector similarity between query and chunk
    - keyword_overlap: Jaccard similarity of token sets
    - entity_match: 1.0 if entity from query appears in chunk
    - topic_match: 1.0 if topic overlaps
    - rank_score: 1/(1+rank) positional bonus

Global (2 features):
    - query_length: normalized token count (capped at 50 tokens)
    - topic_present: 1.0 if topic info available in query
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from rl_memory_retrieval.models import MemoryItem, QueryItem

# Number of features per candidate
FEATURES_PER_CANDIDATE = 5
# Number of global features
GLOBAL_FEATURES = 2
# Max token count for normalization
_MAX_TOKENS = 50


def compute_state_dimension(top_k: int) -> int:
    """Compute the total state dimension for the RL environment.

    Args:
        top_k: Number of candidate chunks per query.

    Returns:
        Total state vector dimension.
    """
    return FEATURES_PER_CANDIDATE * top_k + GLOBAL_FEATURES


def compute_state(
    query: QueryItem,
    query_embedding: np.ndarray,
    candidates: list[MemoryItem],
    candidate_embeddings: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Compute the state feature vector for one query.

    Args:
        query: The query item.
        query_embedding: 1D embedding vector for the query.
        candidates: List of candidate MemoryItems (length <= top_k).
        candidate_embeddings: 2D array of shape (len(candidates), dim).
        top_k: Total number of candidate slots. Pads with zeros if
               len(candidates) < top_k.

    Returns:
        1D numpy array of shape (compute_state_dimension(top_k),).
    """
    dim = compute_state_dimension(top_k)
    state = np.zeros(dim, dtype=np.float32)

    q_emb = query_embedding.reshape(1, -1)

    for i, (candidate, c_emb) in enumerate(
        zip(candidates[:top_k], candidate_embeddings[:top_k])
    ):
        offset = i * FEATURES_PER_CANDIDATE

        # 1. Cosine similarity
        cos_sim = float(
            sk_cosine_similarity(q_emb, c_emb.reshape(1, -1))[0, 0]
        )
        state[offset + 0] = max(0.0, min(1.0, cos_sim))

        # 2. Keyword overlap (Jaccard similarity of token sets)
        state[offset + 1] = _keyword_overlap(query.query, candidate.text)

        # 3. Entity match
        state[offset + 2] = _entity_match(query.entity, candidate.text)

        # 4. Topic match
        state[offset + 3] = _topic_match(query.topic, candidate.topic)

        # 5. Rank score (1 / (1 + rank))
        state[offset + 4] = 1.0 / (1.0 + i)

    # Global features
    global_offset = FEATURES_PER_CANDIDATE * top_k

    # query_length: normalized token count
    tokens = query.query.split()
    state[global_offset + 0] = min(len(tokens) / _MAX_TOKENS, 1.0)

    # topic_present: 1.0 if topic info available
    state[global_offset + 1] = 1.0 if query.topic else 0.0

    return state


def find_top_k_candidates(
    query_embedding: np.ndarray,
    all_embeddings: np.ndarray,
    all_items: list[MemoryItem],
    top_k: int,
) -> tuple[list[MemoryItem], np.ndarray, list[int]]:
    """Find the top-K most similar candidates for a query.

    Args:
        query_embedding: 1D embedding vector for the query.
        all_embeddings: 2D array of shape (N, dim) for all memory items.
        all_items: List of all MemoryItems.
        top_k: Number of candidates to return.

    Returns:
        Tuple of (candidate_items, candidate_embeddings, candidate_indices).
    """
    q = query_embedding.reshape(1, -1)
    sims = sk_cosine_similarity(q, all_embeddings)[0]
    indices = np.argsort(sims)[::-1][:top_k]

    candidates = [all_items[i] for i in indices]
    cand_embs = all_embeddings[indices]

    return candidates, cand_embs, indices.tolist()


def _keyword_overlap(query_text: str, chunk_text: str) -> float:
    """Compute Jaccard similarity of token sets.

    Args:
        query_text: The query string.
        chunk_text: The candidate chunk text.

    Returns:
        Jaccard similarity in [0, 1].
    """
    q_tokens = set(query_text.lower().split())
    c_tokens = set(chunk_text.lower().split())

    if not q_tokens or not c_tokens:
        return 0.0

    intersection = q_tokens & c_tokens
    union = q_tokens | c_tokens

    return len(intersection) / len(union)


def _entity_match(query_entity: str, chunk_text: str) -> float:
    """Check if the query entity appears in the chunk text.

    Args:
        query_entity: Comma-separated entity string from the query.
        chunk_text: The candidate chunk text.

    Returns:
        1.0 if any entity matches, 0.0 otherwise.
    """
    if not query_entity:
        return 0.0

    chunk_lower = chunk_text.lower()
    entities = [e.strip() for e in query_entity.split(",")]

    for entity in entities:
        if entity and entity.lower() in chunk_lower:
            return 1.0

    return 0.0


def _topic_match(query_topic: str, chunk_topic: str) -> float:
    """Check if the query topic overlaps with the chunk topic.

    Args:
        query_topic: Topic string from the query.
        chunk_topic: Topic string from the chunk.

    Returns:
        1.0 if topics overlap, 0.0 otherwise.
    """
    if not query_topic or not chunk_topic:
        return 0.0

    q_topics = {t.strip().lower() for t in query_topic.split(",")}
    c_topics = {t.strip().lower() for t in chunk_topic.split(",")}

    return 1.0 if q_topics & c_topics else 0.0
