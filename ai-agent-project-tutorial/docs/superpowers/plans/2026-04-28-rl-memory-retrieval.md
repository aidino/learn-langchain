# RL Memory Retrieval Pipeline — Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Build a generalizable pipeline that ingests any knowledge source, trains a PPO-based RL retriever, and answers questions — replacing the hardcoded synthetic memory bank from the tutorial.

**Architecture:** Modular Python package (`rl_memory_retrieval/`) with CLI (Click), config (Pydantic Settings + YAML), pluggable embedders, ingestion layer (Crawl4AI + markitdown), LLM query generation, Gymnasium RL environment, PPO training (stable-baselines3), and evaluation. Each component is independently testable.

**Tech Stack:** Python 3.11, Click, Pydantic Settings, PyYAML, OpenAI, Gymnasium, stable-baselines3, Crawl4AI, markitdown, sentence-transformers (optional), langextract (optional), numpy, scikit-learn, pytest

**Ref spec:** `docs/superpowers/specs/2026-04-28-rl-memory-retrieval-design.md`
**Ref original:** `00_rl_agent_long_term_memory_retrieval.py`

---

## Task 1: Project Scaffolding & Config

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `rl_memory_retrieval/__init__.py`
- Create: `rl_memory_retrieval/config.py`
- Create: `rl_memory_retrieval/models.py`
- Test: `tests/test_config.py`

**Step 1: Create `pyproject.toml`**

Use PEP 621 format. Define `[project]` with name `rl-memory-retrieval`, python `>=3.11`, and all dependencies from the spec's dependency tables. Add `[project.scripts]` entry: `rl-retriever = "rl_memory_retrieval.cli:cli"`. Add `[project.optional-dependencies]` for `extraction` (langextract), `local` (sentence-transformers), `plot` (matplotlib), and `dev` (pytest, pytest-cov).

**Step 2: Create `config.yaml`**

Copy the exact YAML from spec section "2. Config System" (lines 108-147). All defaults as specified.

**Step 3: Create `rl_memory_retrieval/models.py`**

Two dataclasses:

```python
from dataclasses import dataclass, field

@dataclass
class MemoryItem:
    memory_id: int
    text: str
    topic: str = ""
    entity: str = ""
    slot: str = ""
    value: str = ""
    metadata: dict = field(default_factory=dict)

@dataclass
class QueryItem:
    query_id: int
    query: str
    gold_memory_id: int
    gold_value: str
    topic: str = ""
    entity: str = ""
```

**Step 4: Create `rl_memory_retrieval/config.py`**

Use `pydantic-settings` with `YamlConfigSettingsSource`. Nested models: `EmbeddingsConfig`, `ExtractionConfig`, `ChunkingConfig`, `QueryGenConfig`, `TrainingConfig`, `EvaluationConfig`. Main `Settings(BaseSettings)` with `model_config = SettingsConfigDict(env_prefix="RL_", env_file=".env", yaml_file="config.yaml")`. Override `settings_customise_sources` to add YAML source. See spec lines 99-147 for all fields and defaults.

**Step 5: Create `rl_memory_retrieval/__init__.py`**

```python
from rl_memory_retrieval.config import Settings
__all__ = ["Settings"]
```

**Step 6: Write test**

```python
# tests/test_config.py
from rl_memory_retrieval.config import Settings

def test_default_settings():
    s = Settings(source="https://example.com")
    assert s.embeddings.backend == "openai"
    assert s.training.top_k_candidates == 8
    assert s.chunking.chunk_size == 300
```

**Step 7: Run test**

```bash
cd /home/dino/Documents/learn-langchain/ai-agent-project-tutorial
.venv/bin/python -m pytest tests/test_config.py -v
```

**Step 8: Install package in dev mode & commit**

```bash
.venv/bin/pip install -e ".[dev]"
git add pyproject.toml config.yaml rl_memory_retrieval/ tests/test_config.py
git commit -m "feat: project scaffolding with config and models"
```

---

## Task 2: Ingestion Layer (Loader + Chunker)

**Files:**
- Create: `rl_memory_retrieval/ingestion/__init__.py`
- Create: `rl_memory_retrieval/ingestion/loader.py`
- Create: `rl_memory_retrieval/ingestion/chunker.py`
- Test: `tests/test_chunker.py`

**Step 1: Write chunker test**

```python
# tests/test_chunker.py
from rl_memory_retrieval.ingestion.chunker import chunk_text

def test_chunk_basic():
    text = "A" * 600
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 300 for c in chunks)

def test_chunk_overlap():
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    # Check overlap exists
    assert chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0]

def test_chunk_short_text():
    chunks = chunk_text("short", chunk_size=300, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "short"
```

**Step 2: Run test, verify FAIL**

```bash
.venv/bin/python -m pytest tests/test_chunker.py -v
```

**Step 3: Implement `chunker.py`**

Recursive character text splitting. Split on `["\n\n", "\n", ". ", " ", ""]` separators. Parameters: `chunk_size`, `chunk_overlap`. Return `list[str]`.

**Step 4: Implement `loader.py`**

```python
import asyncio
from pathlib import Path

def load_source(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        return _load_url(source)
    path = Path(source)
    if path.is_dir():
        return _load_directory(path)
    return _load_file(path)

def _load_url(url: str) -> str:
    from crawl4ai import AsyncWebCrawler
    async def _crawl():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown
    return asyncio.run(_crawl())

def _load_file(path: Path) -> str:
    if path.suffix in (".md", ".txt"):
        return path.read_text()
    from markitdown import MarkItDown
    md = MarkItDown()
    result = md.convert(str(path))
    return result.markdown

def _load_directory(path: Path) -> str:
    texts = []
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix in (".md",".txt",".pdf",".docx",".pptx"):
            texts.append(_load_file(f))
    return "\n\n".join(texts)
```

**Step 5: Create `ingestion/__init__.py`**

```python
from .loader import load_source
from .chunker import chunk_text
```

**Step 6: Run tests, verify PASS**

```bash
.venv/bin/python -m pytest tests/test_chunker.py -v
```

**Step 7: Commit**

```bash
git add rl_memory_retrieval/ingestion/ tests/test_chunker.py
git commit -m "feat: ingestion layer with loader and chunker"
```

---

## Task 3: Embeddings Layer

**Files:**
- Create: `rl_memory_retrieval/embeddings/__init__.py`
- Create: `rl_memory_retrieval/embeddings/base.py`
- Create: `rl_memory_retrieval/embeddings/openai_embedder.py`
- Create: `rl_memory_retrieval/embeddings/local_embedder.py`

**Step 1: Create `base.py` with Protocol**

```python
from typing import Protocol
import numpy as np

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    @property
    def dimension(self) -> int: ...
```

**Step 2: Implement `openai_embedder.py`**

Port `embed_texts()` from original file (lines 85-96). Class `OpenAIEmbedder` with `__init__(self, model, api_key, batch_size)`. Method `embed()` calls OpenAI API, L2 normalizes. Property `dimension` returns vector size from first call or known default (1536 for text-embedding-3-small).

**Step 3: Implement `local_embedder.py`**

Wrap `sentence_transformers.SentenceTransformer`. Class `LocalEmbedder` with `__init__(self, model_name="all-MiniLM-L6-v2")`. Method `embed()` calls `model.encode()`, L2 normalizes.

**Step 4: Create `embeddings/__init__.py`**

Factory function `create_embedder(config) -> Embedder` that returns `OpenAIEmbedder` or `LocalEmbedder` based on `config.embeddings.backend`.

**Step 5: Commit**

```bash
git add rl_memory_retrieval/embeddings/
git commit -m "feat: pluggable embeddings layer (openai + local)"
```

---

## Task 4: Optional Metadata Extractor

**Files:**
- Create: `rl_memory_retrieval/ingestion/extractor.py`

**Step 1: Implement `extractor.py`**

```python
def extract_metadata(text: str, instruction: str, model_id: str, api_key: str) -> dict:
    try:
        import langextract as lx
        result = lx.extract(
            text_or_documents=text,
            prompt_description=instruction,
            model_id=model_id,
            api_key=api_key,
            fence_output=True,
            use_schema_constraints=False,
        )
        return {"entities": result.get("entities", []), "topics": result.get("topics", [])}
    except ImportError:
        return {"entities": [], "topics": []}
```

**Step 2: Commit**

```bash
git add rl_memory_retrieval/ingestion/extractor.py
git commit -m "feat: optional langextract metadata extraction"
```

---

## Task 5: Query Generation

**Files:**
- Create: `rl_memory_retrieval/training/__init__.py`
- Create: `rl_memory_retrieval/training/query_gen.py`
- Test: `tests/test_query_gen.py`

**Step 1: Write test**

```python
# tests/test_query_gen.py
from rl_memory_retrieval.training.query_gen import parse_qa_response

def test_parse_qa_response_valid():
    raw = '[{"question": "What is X?", "answer": "Y"}]'
    pairs = parse_qa_response(raw)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "What is X?"

def test_parse_qa_response_invalid():
    pairs = parse_qa_response("not json")
    assert pairs == []
```

**Step 2: Run test, verify FAIL**

**Step 3: Implement `query_gen.py`**

Functions:
- `parse_qa_response(raw: str) -> list[dict]` — JSON parse with error handling
- `generate_queries_for_chunk(chunk_text, chunk_id, n, model, api_key) -> list[QueryItem]` — calls OpenAI chat with system prompt from spec (lines 218-220), parses response
- `deduplicate_queries(queries, embedder, threshold=0.9) -> list[QueryItem]` — embed all queries, skip if cosine sim > 0.9 to existing

**Step 4: Run tests, verify PASS**

**Step 5: Commit**

```bash
git add rl_memory_retrieval/training/ tests/test_query_gen.py
git commit -m "feat: LLM query generation with deduplication"
```

---

## Task 6: RL Feature Engineering

**Files:**
- Create: `rl_memory_retrieval/training/features.py`
- Test: `tests/test_features.py`

**Step 1: Write test**

```python
# tests/test_features.py
import numpy as np
from rl_memory_retrieval.training.features import build_state_features, keyword_overlap

def test_keyword_overlap():
    assert keyword_overlap("what is battery", "battery life is good") > 0
    assert keyword_overlap("hello", "world") == 0.0

def test_build_state_features_shape():
    candidates = [{"sim": 0.9, "overlap": 0.3, "entity_match": 1.0,
                    "topic_match": 0.0, "rank": i} for i in range(8)]
    query_info = {"query_length": 0.5, "topic_present": 1.0}
    feats = build_state_features(candidates, query_info, top_k=8)
    assert feats.shape == (42,)  # 5*8 + 2
    assert feats.dtype == np.float32
```

**Step 2: Run test, verify FAIL**

**Step 3: Implement `features.py`**

Port `keyword_overlap()` from original (lines 382-389) and `build_state_features()` from original (lines 429-446). Adapt to use the 5 features per candidate from spec (cosine_similarity, keyword_overlap, entity_match, topic_match, rank_score) plus 2 global (query_length, topic_present).

**Step 4: Run tests, verify PASS**

**Step 5: Commit**

```bash
git add rl_memory_retrieval/training/features.py tests/test_features.py
git commit -m "feat: RL state feature engineering"
```

---

## Task 7: Gymnasium RL Environment

**Files:**
- Create: `rl_memory_retrieval/training/environment.py`
- Test: `tests/test_environment.py`

**Step 1: Write test**

```python
# tests/test_environment.py
import numpy as np
from rl_memory_retrieval.training.environment import MemoryRetrievalEnv

def _make_dummy_items(n=5, k=8):
    items = []
    for i in range(n):
        candidates = []
        for j in range(k):
            candidates.append({
                "sim": 0.5 + 0.05*j, "overlap": 0.1, "entity_match": 1.0 if j==0 else 0.0,
                "topic_match": 0.0, "rank": j, "memory_id": j, "is_gold": 1.0 if j==0 else 0.0,
                "text": f"memory {j}",
            })
        items.append({
            "query": {"query_id": i, "query": "test?", "gold_memory_id": 0,
                       "gold_value": "v", "topic": "t", "entity": "e"},
            "candidates": candidates,
        })
    return items

def test_env_reset():
    env = MemoryRetrievalEnv(_make_dummy_items())
    obs, info = env.reset()
    assert obs.shape == (42,)
    assert "query_id" in info

def test_env_step_correct():
    env = MemoryRetrievalEnv(_make_dummy_items())
    env.reset()
    obs, reward, done, truncated, info = env.step(0)
    assert done is True
    assert reward > 0
    assert info["is_correct"] is True

def test_env_step_wrong():
    env = MemoryRetrievalEnv(_make_dummy_items())
    env.reset()
    _, reward_wrong, _, _, info = env.step(7)
    env.reset()
    _, reward_right, _, _, _ = env.step(0)
    assert reward_right > reward_wrong
```

**Step 2: Run test, verify FAIL**

**Step 3: Implement `environment.py`**

Port `MemoryRetrievalEnv` from original (lines 453-506). Use the reward function from spec (lines 256-261). State dim = `5 * top_k + 2`. Action space = `Discrete(top_k)`. One-step episodes.

**Step 4: Run tests, verify PASS**

**Step 5: Commit**

```bash
git add rl_memory_retrieval/training/environment.py tests/test_environment.py
git commit -m "feat: Gymnasium RL environment for memory retrieval"
```

---

## Task 8: PPO Trainer

**Files:**
- Create: `rl_memory_retrieval/training/trainer.py`

**Step 1: Implement `trainer.py`**

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_memory_retrieval.training.environment import MemoryRetrievalEnv
from rl_memory_retrieval.config import Settings
import numpy as np
from pathlib import Path

def train_model(train_items, config: Settings, seed: int = 42) -> PPO:
    train_env = DummyVecEnv([lambda: MemoryRetrievalEnv(train_items, seed=seed)])
    tc = config.training
    model = PPO("MlpPolicy", train_env,
                learning_rate=tc.learning_rate, n_steps=tc.n_steps,
                batch_size=tc.batch_size, gamma=tc.gamma,
                gae_lambda=tc.gae_lambda, ent_coef=tc.ent_coef,
                clip_range=tc.clip_range, verbose=0, seed=seed)
    model.learn(total_timesteps=tc.total_timesteps)
    return model

def save_model(model: PPO, embeddings: np.ndarray, output_dir: str):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path / "ppo_model"))
    np.save(str(path / "embeddings.npy"), embeddings)
```

**Step 2: Commit**

```bash
git add rl_memory_retrieval/training/trainer.py
git commit -m "feat: PPO trainer with configurable hyperparameters"
```

---

## Task 9: Retrieval & Evaluation

**Files:**
- Create: `rl_memory_retrieval/retrieval/__init__.py`
- Create: `rl_memory_retrieval/retrieval/retriever.py`
- Create: `rl_memory_retrieval/retrieval/evaluator.py`
- Test: `tests/test_retriever.py`

**Step 1: Write test**

```python
# tests/test_retriever.py
from unittest.mock import MagicMock
import numpy as np
from rl_memory_retrieval.retrieval.retriever import baseline_retrieve

def test_baseline_retrieve():
    item = {"candidates": [
        {"sim": 0.3, "memory_id": 1, "text": "a"},
        {"sim": 0.9, "memory_id": 2, "text": "b"},
        {"sim": 0.5, "memory_id": 3, "text": "c"},
    ]}
    chosen = baseline_retrieve(item)
    assert chosen["memory_id"] == 2  # highest sim
```

**Step 2: Run test, verify FAIL**

**Step 3: Implement `retriever.py`**

Port `baseline_retrieve()` and `rl_retrieve()` from original (lines 535-543). `rl_retrieve` takes a PPO model and uses `model.predict(obs, deterministic=True)`.

**Step 4: Implement `evaluator.py`**

Port `evaluate_retriever()` from original (lines 546-564). Add optional LLM judge evaluation using `chat_answer()` and `llm_judge_exact()` from original (lines 99-140).

**Step 5: Run tests, verify PASS**

**Step 6: Commit**

```bash
git add rl_memory_retrieval/retrieval/ tests/test_retriever.py
git commit -m "feat: retrieval and evaluation with baseline comparison"
```

---

## Task 10: Pipeline Orchestrator

**Files:**
- Create: `rl_memory_retrieval/pipeline.py`
- Update: `rl_memory_retrieval/__init__.py`

**Step 1: Implement `pipeline.py`**

Class `Pipeline` that orchestrates the full flow from spec (lines 26-38):

```python
class Pipeline:
    def __init__(self, config: Settings): ...

    @classmethod
    def from_config(cls, config_path: str) -> "Pipeline": ...

    def train(self, source: str, output_dir: str) -> dict:
        # 1. load_source(source)
        # 2. chunk_text(content, ...)
        # 3. Create MemoryItems with IDs
        # 4. Optional: extract_metadata per chunk
        # 5. Embed chunks
        # 6. Generate queries from chunks
        # 7. Deduplicate queries
        # 8. Build candidate sets (top-K per query)
        # 9. Split train/val/test
        # 10. Train PPO
        # 11. Evaluate on val+test
        # 12. Save model + embeddings + metadata
        ...

    def query(self, question: str, model_dir: str) -> dict:
        # Load saved model + embeddings
        # Embed question, find top-K, build features
        # RL predict, return answer
        ...
```

**Step 2: Update `__init__.py`**

```python
from rl_memory_retrieval.pipeline import Pipeline
from rl_memory_retrieval.config import Settings
__all__ = ["Pipeline", "Settings"]
```

**Step 3: Commit**

```bash
git add rl_memory_retrieval/pipeline.py rl_memory_retrieval/__init__.py
git commit -m "feat: pipeline orchestrator for train and query"
```

---

## Task 11: CLI Interface

**Files:**
- Create: `rl_memory_retrieval/cli.py`

**Step 1: Implement `cli.py`**

```python
import click
from rl_memory_retrieval.pipeline import Pipeline
from rl_memory_retrieval.config import Settings

@click.group()
def cli():
    """RL Memory Retrieval — Train and query RL-powered memory retrieval."""
    pass

@cli.command()
@click.option("--source", required=True, help="URL, file, or directory")
@click.option("--output", default="./model_output", help="Output directory")
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
def train(source, output, config_path):
    """Train a new RL retriever from a knowledge source."""
    if config_path:
        pipeline = Pipeline.from_config(config_path)
    else:
        pipeline = Pipeline(Settings(source=source))
    result = pipeline.train(source=source, output_dir=output)
    click.echo(f"Training complete. Results: {result}")

@cli.command()
@click.option("--model", required=True, help="Path to trained model dir")
@click.option("--question", required=True, help="Question to answer")
def query(model, question):
    """Query a trained RL retriever."""
    pipeline = Pipeline(Settings())
    result = pipeline.query(question=question, model_dir=model)
    click.echo(f"Answer: {result['answer']}")

if __name__ == "__main__":
    cli()
```

**Step 2: Commit**

```bash
git add rl_memory_retrieval/cli.py
git commit -m "feat: Click CLI with train and query commands"
```

---

## Task 12: Integration Test & Final Verification

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test with small synthetic data (no real API calls)."""
from unittest.mock import patch, MagicMock
import numpy as np
from rl_memory_retrieval.ingestion.chunker import chunk_text
from rl_memory_retrieval.training.features import build_state_features, keyword_overlap
from rl_memory_retrieval.training.environment import MemoryRetrievalEnv
from rl_memory_retrieval.retrieval.retriever import baseline_retrieve

def test_end_to_end_chunking_to_retrieval():
    text = "Astra robot has 18 hour battery. " * 20
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2

    # Simulate candidates
    candidates = [{"sim": 0.9-i*0.1, "overlap": 0.2, "entity_match": 1.0 if i==0 else 0.0,
                    "topic_match": 0.0, "rank": i, "memory_id": i, "is_gold": 1.0 if i==0 else 0.0,
                    "text": chunks[i % len(chunks)]} for i in range(8)]
    item = {"query": {"query_id": 0, "query": "What is battery?", "gold_memory_id": 0,
                       "gold_value": "18h", "topic": "robotics", "entity": "Astra"},
            "candidates": candidates}

    chosen = baseline_retrieve(item)
    assert chosen["memory_id"] == 0
```

**Step 2: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

**Step 3: Verify CLI installs**

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/rl-retriever --help
```

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration test for end-to-end flow"
```
