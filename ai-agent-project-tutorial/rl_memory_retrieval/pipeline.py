"""Pipeline orchestrator for the RL Memory Retrieval system.

Wires together all modules in the correct order:
    Ingest → Chunk → [Extract] → Embed → Generate Queries → Build RL Dataset
    → Train PPO → Evaluate → Save Model

Provides a simple API:
    pipeline.train(source, output_dir) → metrics dict
    pipeline.query(question, model_dir) → answer dict
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rl_memory_retrieval.config import Settings
from rl_memory_retrieval.embeddings import create_embedder
from rl_memory_retrieval.ingestion import chunk_text, load_source
from rl_memory_retrieval.models import MemoryItem, QueryItem
from rl_memory_retrieval.retrieval.evaluator import compare_retrievers
from rl_memory_retrieval.retrieval.retriever import baseline_retrieve, rl_retrieve
from rl_memory_retrieval.training.features import (
    _keyword_overlap,
    _entity_match,
    _topic_match,
    find_top_k_candidates,
)
from rl_memory_retrieval.training.trainer import load_model, save_model, train_model

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end pipeline for RL-powered memory retrieval.

    Args:
        config: Application settings. If not provided, uses defaults.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or Settings()

    @classmethod
    def from_config(cls, config_path: str) -> "Pipeline":
        """Create a Pipeline from a YAML config file.

        Args:
            config_path: Path to a YAML config file.

        Returns:
            A Pipeline instance with the loaded config.
        """
        settings = Settings(_yaml_file=config_path)
        return cls(config=settings)

    def train(self, source: str, output_dir: str | None = None) -> dict:
        """Train an RL retriever from a knowledge source.

        Steps:
            1. Load content from source (URL, file, or directory)
            2. Chunk content into memory-sized pieces
            3. Create MemoryItems with unique IDs
            4. Optionally extract metadata per chunk
            5. Embed all chunks
            6. Generate QA pairs from chunks via LLM
            7. Deduplicate queries
            8. Build candidate sets (top-K per query)
            9. Split into train/val/test
            10. Train PPO model
            11. Evaluate on val + test
            12. Save model, embeddings, and metadata

        Args:
            source: URL, file path, or directory path to ingest.
            output_dir: Directory to save trained model artifacts.
                        Defaults to config.output_dir.

        Returns:
            Dict with training results and evaluation metrics.
        """
        output_dir = output_dir or self.config.output_dir
        cfg = self.config
        top_k = cfg.training.top_k_candidates

        # ----- Step 1: Ingest -----
        logger.info("Step 1: Loading source: %s", source)
        content = load_source(source)
        logger.info("Loaded %d characters", len(content))

        # ----- Step 2: Chunk -----
        logger.info("Step 2: Chunking content")
        chunks = chunk_text(
            content,
            chunk_size=cfg.chunking.chunk_size,
            chunk_overlap=cfg.chunking.chunk_overlap,
        )
        logger.info("Created %d chunks", len(chunks))

        if len(chunks) < 2:
            raise ValueError(
                f"Source produced only {len(chunks)} chunks. "
                "Need at least 2 chunks for training."
            )

        # ----- Step 3: Create MemoryItems -----
        logger.info("Step 3: Creating MemoryItems")
        items = [
            MemoryItem(memory_id=i, text=chunk)
            for i, chunk in enumerate(chunks)
        ]

        # ----- Step 4: Optional metadata extraction -----
        if cfg.extraction.enabled:
            logger.info("Step 4: Extracting metadata")
            try:
                from rl_memory_retrieval.ingestion import extract_metadata

                items = extract_metadata(
                    items,
                    instruction=cfg.extraction.instruction,
                    model_id=cfg.query_generation.model,
                )
            except ImportError:
                logger.warning(
                    "langextract not installed — skipping metadata extraction"
                )
        else:
            logger.info("Step 4: Metadata extraction disabled — skipping")

        # ----- Step 5: Embed chunks -----
        logger.info("Step 5: Embedding %d chunks", len(items))
        embedder = create_embedder(cfg.embeddings)
        texts = [item.text for item in items]
        embeddings = embedder.embed(texts)
        logger.info("Embedding shape: %s", embeddings.shape)

        # ----- Step 6: Generate queries -----
        logger.info("Step 6: Generating queries from chunks")
        from rl_memory_retrieval.training.query_gen import generate_queries

        queries = generate_queries(
            items,
            queries_per_chunk=cfg.query_generation.queries_per_chunk,
            model=cfg.query_generation.model,
            embedder=embedder if True else None,
            dedup_threshold=0.9,
        )
        logger.info("Generated %d queries (after dedup)", len(queries))

        if len(queries) < 3:
            raise ValueError(
                f"Only {len(queries)} queries generated. "
                "Need at least 3 for train/val/test split."
            )

        # ----- Step 7: Build candidate sets -----
        logger.info("Step 7: Building candidate sets (top-%d per query)", top_k)
        query_texts = [q.query for q in queries]
        query_embeddings = embedder.embed(query_texts)

        rl_items = []
        for qi, q_emb in tqdm(
            zip(queries, query_embeddings),
            total=len(queries),
            desc="Building candidates",
        ):
            candidates, cand_embs, indices = find_top_k_candidates(
                q_emb, embeddings, items, top_k
            )

            candidate_dicts = []
            for rank, (cand, c_emb, idx) in enumerate(
                zip(candidates, cand_embs, indices)
            ):
                from sklearn.metrics.pairwise import cosine_similarity as sk_cos

                sim = float(sk_cos(q_emb.reshape(1, -1), c_emb.reshape(1, -1))[0, 0])
                overlap = _keyword_overlap(qi.query, cand.text)
                entity_match = _entity_match(qi.entity, cand.text)
                topic_match = _topic_match(qi.topic, cand.topic)

                candidate_dicts.append({
                    "sim": sim,
                    "overlap": overlap,
                    "entity_match": entity_match,
                    "topic_match": topic_match,
                    "rank": rank,
                    "memory_id": cand.memory_id,
                    "is_gold": 1.0 if cand.memory_id == qi.gold_memory_id else 0.0,
                    "text": cand.text,
                })

            rl_items.append({
                "query": {
                    "query_id": qi.query_id,
                    "query": qi.query,
                    "gold_memory_id": qi.gold_memory_id,
                    "gold_value": qi.gold_value,
                    "topic": qi.topic,
                    "entity": qi.entity,
                },
                "candidates": candidate_dicts,
            })

        # ----- Step 8: Split train/val/test -----
        logger.info("Step 8: Splitting data")
        n = len(rl_items)
        train_end = int(n * cfg.training.train_split)
        val_end = train_end + int(n * cfg.training.val_split)

        # Shuffle with a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        rl_items_shuffled = [rl_items[i] for i in indices]

        train_items = rl_items_shuffled[:train_end]
        val_items = rl_items_shuffled[train_end:val_end]
        test_items = rl_items_shuffled[val_end:]

        logger.info(
            "Split: %d train, %d val, %d test",
            len(train_items), len(val_items), len(test_items),
        )

        if len(train_items) == 0:
            raise ValueError("No training items after split. Need more data.")

        # ----- Step 9: Train PPO -----
        logger.info("Step 9: Training PPO model")
        model = train_model(train_items, self.config)

        # ----- Step 10: Evaluate -----
        logger.info("Step 10: Evaluating model")
        eval_items = val_items + test_items if test_items else val_items

        if eval_items:
            results = compare_retrievers(
                eval_items,
                {
                    "baseline": baseline_retrieve,
                    "rl": partial(rl_retrieve, model=model, top_k=top_k),
                },
            )
        else:
            results = {"baseline": {"accuracy": 0.0}, "rl": {"accuracy": 0.0}}

        logger.info(
            "Baseline accuracy: %.2f%%, RL accuracy: %.2f%%",
            results["baseline"]["accuracy"] * 100,
            results["rl"]["accuracy"] * 100,
        )

        # ----- Step 11: Save -----
        logger.info("Step 11: Saving model to %s", output_dir)
        metadata = {
            "source": source,
            "num_chunks": len(items),
            "num_queries": len(queries),
            "num_train": len(train_items),
            "num_val": len(val_items),
            "num_test": len(test_items),
            "baseline_accuracy": results["baseline"]["accuracy"],
            "rl_accuracy": results["rl"]["accuracy"],
            "config": {
                "chunk_size": cfg.chunking.chunk_size,
                "chunk_overlap": cfg.chunking.chunk_overlap,
                "top_k": top_k,
                "total_timesteps": cfg.training.total_timesteps,
                "embeddings_backend": cfg.embeddings.backend,
                "embeddings_model": cfg.embeddings.model,
            },
        }

        # Save memory items for query-time retrieval
        items_data = [asdict(item) for item in items]

        save_model(model, embeddings, output_dir, metadata=metadata)

        # Save items separately for query-time use
        items_path = Path(output_dir) / "items.json"
        with open(items_path, "w") as f:
            json.dump(items_data, f, indent=2)

        logger.info("Training complete!")

        return {
            "num_chunks": len(items),
            "num_queries": len(queries),
            "splits": {
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
            },
            "evaluation": results,
        }

    def query(self, question: str, model_dir: str) -> dict:
        """Query a trained RL retriever.

        Loads the saved model and embeddings, embeds the question,
        finds top-K candidates, and uses the RL model to select the best.

        Args:
            question: The question to answer.
            model_dir: Path to the directory containing saved model artifacts.

        Returns:
            Dict with 'answer', 'memory_id', 'text', 'sim', and 'method'.
        """
        top_k = self.config.training.top_k_candidates

        # Load saved model and embeddings
        model, embeddings, metadata = load_model(model_dir)

        # Load saved memory items
        items_path = Path(model_dir) / "items.json"
        if items_path.exists():
            with open(items_path) as f:
                items_data = json.load(f)
            items = [MemoryItem(**d) for d in items_data]
        else:
            raise FileNotFoundError(
                f"No items.json found in {model_dir}. "
                "Was the model trained with this version of the pipeline?"
            )

        # Embed the question
        embedder = create_embedder(self.config.embeddings)
        q_embedding = embedder.embed([question])[0]

        # Find top-K candidates
        candidates, cand_embs, indices = find_top_k_candidates(
            q_embedding, embeddings, items, top_k
        )

        # Build candidate dicts with features
        candidate_dicts = []
        for rank, (cand, c_emb) in enumerate(zip(candidates, cand_embs)):
            from sklearn.metrics.pairwise import cosine_similarity as sk_cos

            sim = float(
                sk_cos(q_embedding.reshape(1, -1), c_emb.reshape(1, -1))[0, 0]
            )
            overlap = _keyword_overlap(question, cand.text)
            entity_match = _entity_match("", cand.text)
            topic_match = _topic_match("", cand.topic)

            candidate_dicts.append({
                "sim": sim,
                "overlap": overlap,
                "entity_match": entity_match,
                "topic_match": topic_match,
                "rank": rank,
                "memory_id": cand.memory_id,
                "is_gold": 0.0,  # Unknown at query time
                "text": cand.text,
            })

        # Use RL model to select best candidate
        item = {
            "query": {
                "query_id": 0,
                "query": question,
                "gold_memory_id": -1,
                "gold_value": "",
                "topic": "",
                "entity": "",
            },
            "candidates": candidate_dicts,
        }

        chosen = rl_retrieve(item, model=model, top_k=top_k)

        return {
            "answer": chosen["text"],
            "memory_id": chosen["memory_id"],
            "text": chosen["text"],
            "sim": chosen["sim"],
            "method": "rl",
        }
