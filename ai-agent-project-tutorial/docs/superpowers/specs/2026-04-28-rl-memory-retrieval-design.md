# RL Memory Retrieval Pipeline — Design Spec

**Date:** 2026-04-28
**Status:** Draft
**Author:** Dino + Claude

## Problem

The tutorial "Build a Reinforcement Learning Powered Agent that Learns to Retrieve Relevant Long-Term Memories for Accurate LLM Question Answering" demonstrates a PPO-based RL agent that learns to select the best memory from candidate chunks. However, the code is tightly coupled to a hardcoded synthetic memory bank — it cannot ingest real documents, URLs, or PDFs.

We need a generalizable, reusable pipeline that takes any knowledge source (URL, PDF, DOCX, etc.), builds a memory bank, trains an RL retriever, and answers questions accurately.

## Scope

- **In**: Pipeline for ingesting knowledge, training RL retrieval model, querying
- **In**: CLI tool and Python library interface
- **In**: Pluggable embedding backends (OpenAI, local sentence-transformers)
- **In**: LLM-generated training queries
- **In**: Optional structured metadata extraction (langextract)
- **Out**: Distributed training, multiple RL algorithms (PPO only), full IR benchmark suite

## Architecture

### Pipeline Flow

```
Source (URL/PDF/DOC) → Ingest → Chunk → [Extract Metadata] → Embed → Generate Queries → Build RL Dataset → Train PPO → Evaluate → Save Model
```

1. **Ingest** — Load knowledge from source (Crawl4AI for URLs, markitdown for files)
2. **Chunk** — Split content into memory-sized pieces (~300 chars, ~50 overlap)
3. **Extract** (optional) — langextract pulls entities/topics from chunks
4. **Embed** — Convert chunks to vectors via pluggable embedder
5. **Generate Queries** — LLM generates QA pairs from chunks
6. **Build RL Dataset** — For each query, find top-K candidates, compute features
7. **Train** — PPO agent learns to pick the best chunk per query
8. **Evaluate** — Compare RL retriever vs cosine baseline
9. **Save** — Persist model, embeddings, metadata

### Project Structure

```
ai-agent-project-tutorial/
├── .env
├── .python-version
├── pyproject.toml
├── config.yaml                  # Default configuration
├── rl_memory_retrieval/         # Main package
│   ├── __init__.py              # Public API: Pipeline, query
│   ├── cli.py                   # Click CLI entry point
│   ├── config.py                # Pydantic Settings (yaml + env + cli)
│   ├── pipeline.py              # Pipeline orchestrator class
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py            # Unified loader (URL, file, directory)
│   │   ├── chunker.py           # Recursive character text chunking
│   │   └── extractor.py         # Optional langextract metadata
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py              # Embedder protocol
│   │   ├── openai_embedder.py   # OpenAI text-embedding-3-small
│   │   └── local_embedder.py    # sentence-transformers wrapper
│   ├── training/
│   │   ├── __init__.py
│   │   ├── environment.py       # Gymnasium RL env (MemoryRetrievalEnv)
│   │   ├── features.py          # State feature engineering
│   │   ├── query_gen.py         # LLM QA pair generation
│   │   └── trainer.py           # PPO training loop
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py         # RL + baseline retrievers
│   │   └── evaluator.py         # Evaluation & comparison
│   └── models.py                # MemoryItem, QueryItem dataclasses
└── tests/
    ├── test_chunker.py
    ├── test_features.py
    ├── test_environment.py
    ├── test_query_gen.py
    └── test_retriever.py
```

## Components

### 1. CLI Interface

```bash
# Train a new model from a knowledge source
rl-retriever train --source <url|file|dir> --output ./model_output

# Query a trained model
rl-retriever query --model ./model_output --question "What is X?"

# Override config values
rl-retriever train --source ./docs --output ./model --config config.yaml
```

Built with Click. Each subcommand maps to a Pipeline method.

### 2. Config System (Pydantic Settings)

Loads from (priority order):
1. CLI arguments (highest)
2. Environment variables (`RL_` prefix)
3. `.env` file
4. `config.yaml` file
5. Defaults (lowest)

```yaml
# config.yaml
source: null                    # URL or file path (required via CLI)
output_dir: ./model_output

embeddings:
  backend: openai               # "openai" or "local"
  model: text-embedding-3-small # or "all-MiniLM-L6-v2"
  batch_size: 64

extraction:
  enabled: false                # langextract metadata extraction
  instruction: "Extract entities, topics, and key facts"

chunking:
  chunk_size: 300               # characters
  chunk_overlap: 50             # characters

query_generation:
  queries_per_chunk: 3
  model: gpt-4o-mini

training:
  total_timesteps: 12000
  learning_rate: 3e-4
  n_steps: 256
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95
  ent_coef: 0.01
  clip_range: 0.2
  top_k_candidates: 8
  train_split: 0.70
  val_split: 0.15
  # test_split: 0.15 (remainder)

evaluation:
  sample_size: 20               # queries for LLM-judged QA eval
  use_llm_judge: true
```

### 3. Ingestion Layer

**Loader** (`ingestion/loader.py`):

| Source | Tool | API |
|--------|------|-----|
| URL/website | Crawl4AI | `AsyncWebCrawler.arun(url)` → `result.markdown` |
| PDF/DOCX/PPTX/XLSX/etc. | markitdown | `MarkItDown().convert(path)` → `result.markdown` |
| Local .md/.txt | Direct read | `Path.read_text()` |

Source detection by prefix:
- `http://` or `https://` → Crawl4AI
- File extension → markitdown or direct read
- Directory → iterate over files

**Chunker** (`ingestion/chunker.py`):

Recursive character text splitting. Parameters from config:
- `chunk_size`: 300 characters (tunable)
- `chunk_overlap`: 50 characters (tunable)

Each chunk becomes a `MemoryItem` with a unique ID and the original text.

**Extractor** (`ingestion/extractor.py`) — Optional:

Uses langextract to extract structured metadata per chunk:

```python
import langextract as lx

result = lx.extract(
    text_or_documents=chunk_text,
    prompt_description=extraction_instruction,
    model_id="gpt-4o-mini",
    api_key=api_key,
    fence_output=True,
    use_schema_constraints=False,
)
```

Extracted entities and topics are stored as metadata on `MemoryItem`. When extraction is disabled, entity/topic features in the RL environment default to 0.0.

### 4. Embeddings (`embeddings/`)

**Protocol:**

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    @property
    def dimension(self) -> int: ...
```

**OpenAI implementation** (`openai_embedder.py`):
- Model: `text-embedding-3-small` (configurable)
- Batching via `chunked()` helper
- L2 normalization post-embedding
- API key from config/env

**Local implementation** (`local_embedder.py`):
- Wraps `sentence-transformers` (`all-MiniLM-L6-v2` default)
- No API key needed, runs offline
- Same protocol interface

### 5. Query Generation (`training/query_gen.py`)

LLM generates QA pairs from each knowledge chunk:

```python
system_prompt = """Generate {n} diverse question-answer pairs from this text.
Each question should require information from the text to answer.
Return JSON array of {{"question": ..., "answer": ...}}"""
```

Parameters:
- `queries_per_chunk`: 3 (configurable)
- Model: `gpt-4o-mini` (configurable)
- Temperature: 0.7 for diversity

**Deduplication:** After generation, compute embeddings of all queries and skip any query with cosine similarity > 0.9 to an existing query. This prevents near-duplicate training examples.

Each generated query gets:
- `gold_memory_id`: the chunk ID it was generated from
- `gold_value`: the expected answer text
- `topic` / `entity`: from extraction metadata (if available)

### 6. RL Training Environment (`training/environment.py`)

**State features** (per candidate, 5 features × K candidates + 2 global):

| Feature | Description | Range |
|---------|-------------|-------|
| `cosine_similarity` | Vector similarity between query and chunk | [0, 1] |
| `keyword_overlap` | Jaccard similarity of token sets | [0, 1] |
| `entity_match` | 1.0 if any extracted entity from query appears in chunk | {0.0, 1.0} |
| `topic_match` | 1.0 if extracted topic overlaps | {0.0, 1.0} |
| `rank_score` | 1/(1+rank) positional bonus | (0, 1] |
| `query_length` | Normalized token count | [0, 1] |
| `topic_present` | 1.0 if topic info available in query | {0.0, 1.0} |

State dimension: `5 × top_k_candidates + 2`. Default: `5 × 8 + 2 = 42`.

**Action space:** `Discrete(top_k_candidates)` — select one candidate chunk.

**Reward function:**

```python
reward = 2.0 * is_gold           # Correct chunk = big reward
       + 0.8 * entity_match      # Entity matches
       + 0.6 * topic_match       # Topic matches
       + 0.5 * cosine_sim        # High similarity bonus
       + 0.3 * keyword_overlap   # Keyword overlap bonus
       - 0.15 * rank             # Rank penalty
```

Episode structure: one-step (single action per episode). This matches the tutorial design where each query is an independent decision.

**Dataset split:** 70% train, 15% validation, 15% test (configurable).

### 7. PPO Trainer (`training/trainer.py`)

Uses stable-baselines3 PPO with configurable hyperparameters:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

train_env = DummyVecEnv([lambda: MemoryRetrievalEnv(train_items, seed=seed)])
model = PPO("MlpPolicy", train_env, **ppo_kwargs, seed=seed)
model.learn(total_timesteps=config.training.total_timesteps)
```

All PPO hyperparameters exposed in config (learning_rate, n_steps, batch_size, gamma, gae_lambda, ent_coef, clip_range).

Model saved via `model.save()` and embeddings via `np.save()`.

### 8. Retrieval & Evaluation (`retrieval/`)

**Baseline retriever:** Select candidate with highest cosine similarity (same as tutorial).

**RL retriever:** Feed state features to trained PPO model, use `model.predict(deterministic=True)`.

**Evaluator** compares both on:
- **Retrieval accuracy:** `chosen_memory_id == gold_memory_id`
- **QA accuracy (optional):** LLM judge scores whether the answer derived from the retrieved chunk is correct

Print results and optionally plot bar charts.

## Dependencies

### Core

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | >=1.40.0 | Embeddings + chat (query gen, LLM judge) |
| `gymnasium` | >=0.29.1 | RL environment interface |
| `stable-baselines3` | >=2.3.2 | PPO implementation |
| `numpy` | >=1.26.4 | Numerical computation |
| `pandas` | >=2.2.2 | Data manipulation |
| `scikit-learn` | >=1.5.1 | Cosine similarity |
| `pydantic` | >=2.0 | Data validation |
| `pydantic-settings` | >=2.0 | Config from yaml + env |
| `python-dotenv` | >=1.0.0 | .env loading |
| `click` | >=8.0 | CLI framework |
| `tqdm` | >=4.66.4 | Progress bars |

### Ingestion

| Package | Version | Purpose |
|---------|---------|---------|
| `markitdown[pdf,docx,pptx]` | latest | Convert PDF/DOCX/PPTX to markdown |
| `crawl4ai` | latest | Web crawling to markdown |

### Optional

| Package | Version | Purpose |
|---------|---------|---------|
| `langextract` | latest | Structured metadata extraction from chunks |
| `sentence-transformers` | latest | Local embedding model |
| `matplotlib` | >=3.9.0 | Evaluation plots |

### Dev

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | latest | Testing |
| `pytest-cov` | latest | Coverage |

## Python Library Usage

```python
from rl_memory_retrieval import Pipeline
from rl_memory_retrieval.config import Settings

# From config file
pipeline = Pipeline.from_config("config.yaml")
pipeline.train(source="https://example.com/docs", output_dir="./model")

# Query
result = pipeline.query("What is the battery life of Pulse?")
print(result.answer)
```

## Testing Strategy

- **Unit tests:** chunker, feature engineering, environment step/reset, query parsing
- **Integration tests:** full pipeline with small synthetic input
- **No RL training in tests** (too slow) — mock the PPO model for retriever tests
- Target: 80% coverage

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Crawl4AI async complexity | Wrap in sync helper with `asyncio.run()` |
| langextract adds cost per chunk | Make extraction optional, default off |
| LLM query generation quality | Deduplication + temperature tuning; manual spot checks |
| RL training instability | Fixed seed, reasonable defaults, validation monitoring |
| Large knowledge sources | Batch embedding, chunked query generation, progress bars |
