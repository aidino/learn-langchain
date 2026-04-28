# RL Memory Retrieval Pipeline

A generalizable pipeline that ingests any knowledge source (URLs, PDFs, DOCX, etc.), trains a **PPO-based Reinforcement Learning retriever**, and answers questions — replacing hardcoded synthetic memory banks with real-world knowledge.

## Overview

Traditional RAG systems rely on cosine similarity to retrieve relevant chunks. This project trains an RL agent (PPO) that **learns** to pick the best memory chunk by considering multiple signals — not just vector similarity, but also keyword overlap, entity matching, topic alignment, and positional ranking.

### Key Features

- **Any knowledge source** — Ingest from URLs (Crawl4AI), PDFs, DOCX, PPTX (markitdown), or plain text files
- **Pluggable embeddings** — OpenAI `text-embedding-3-small` or local `sentence-transformers`
- **LLM-generated training data** — Automatically creates QA pairs from your content
- **PPO-trained retriever** — Outperforms cosine-similarity baselines on retrieval accuracy
- **Optional metadata extraction** — Uses `langextract` for entity/topic enrichment
- **CLI + Python API** — Use as a command-line tool or import as a library

### Project Status

All core modules have been fully implemented according to the 12-task development plan:
- ✅ Project Scaffolding & Config
- ✅ Ingestion Layer (Loader + Chunker)
- ✅ Embeddings Layer
- ✅ Optional Metadata Extractor
- ✅ Query Generation
- ✅ RL Feature Engineering
- ✅ Gymnasium RL Environment
- ✅ PPO Trainer
- ✅ Retrieval & Evaluation
- ✅ Pipeline Orchestrator
- ✅ CLI Interface
- ✅ Integration Test & Final Verification

## Architecture

```
Source (URL/PDF/DOC)
        │
        ▼
   ┌─────────┐
   │  Ingest  │  ← Crawl4AI (URLs) / markitdown (files)
   └────┬─────┘
        │
        ▼
   ┌─────────┐
   │  Chunk   │  ← Recursive character splitting (~300 chars, ~50 overlap)
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ Extract   │  ← (Optional) langextract for entities/topics
   └────┬──────┘
        │
        ▼
   ┌─────────┐
   │  Embed   │  ← OpenAI or local sentence-transformers
   └────┬─────┘
        │
        ▼
   ┌───────────────┐
   │ Generate QA    │  ← LLM creates question-answer pairs per chunk
   └────┬──────────┘
        │
        ▼
   ┌──────────────────┐
   │ Build RL Dataset  │  ← Top-K candidates per query with feature vectors
   └────┬─────────────┘
        │
        ▼
   ┌───────────┐
   │ Train PPO  │  ← Gymnasium env + stable-baselines3
   └────┬──────┘
        │
        ▼
   ┌───────────┐
   │ Evaluate   │  ← RL vs cosine baseline comparison
   └────┬──────┘
        │
        ▼
   ┌───────────┐
   │   Save     │  ← Model + embeddings + metadata
   └───────────┘
```

### Project Structure

```
ai-agent-project-tutorial/
├── config.yaml                     # Default configuration
├── pyproject.toml                  # PEP 621 project definition
├── rl_memory_retrieval/            # Main package
│   ├── __init__.py                 # Public API: Pipeline, Settings
│   ├── cli.py                      # Click CLI (train / query)
│   ├── config.py                   # Pydantic Settings (yaml + env + cli)
│   ├── models.py                   # MemoryItem, QueryItem dataclasses
│   ├── pipeline.py                 # Pipeline orchestrator
│   ├── ingestion/
│   │   ├── loader.py               # Unified loader (URL, file, directory)
│   │   ├── chunker.py              # Recursive character text chunking
│   │   └── extractor.py            # Optional langextract metadata
│   ├── embeddings/
│   │   ├── base.py                 # Embedder protocol
│   │   ├── openai_embedder.py      # OpenAI text-embedding-3-small
│   │   └── local_embedder.py       # sentence-transformers wrapper
│   ├── training/
│   │   ├── environment.py          # Gymnasium RL env (MemoryRetrievalEnv)
│   │   ├── features.py             # State feature engineering
│   │   ├── query_gen.py            # LLM QA pair generation
│   │   └── trainer.py              # PPO training loop
│   └── retrieval/
│       ├── retriever.py            # RL + baseline retrievers
│       └── evaluator.py            # Evaluation & comparison
└── tests/
    ├── test_config.py
    ├── test_chunker.py
    ├── test_features.py
    ├── test_environment.py
    ├── test_query_gen.py
    ├── test_retriever.py
    └── test_integration.py
```

### RL Environment Design

The RL agent observes a **state vector** (42 dimensions by default) composed of:

| Feature (per candidate × 8) | Description |
|------------------------------|-------------|
| `cosine_similarity` | Vector similarity between query and chunk |
| `keyword_overlap` | Jaccard similarity of token sets |
| `entity_match` | 1.0 if extracted entity appears in chunk |
| `topic_match` | 1.0 if topic overlaps |
| `rank_score` | 1/(1+rank) positional bonus |

Plus 2 global features: `query_length` and `topic_present`.

**Reward function:**
```
reward = 2.0 × is_gold           # Correct chunk
       + 0.8 × entity_match      # Entity matches
       + 0.6 × topic_match       # Topic matches
       + 0.5 × cosine_sim        # High similarity
       + 0.3 × keyword_overlap   # Keyword overlap
       - 0.15 × rank             # Rank penalty
```

## Setup

### Prerequisites

- Python ≥ 3.11
- An OpenAI API key (for embeddings and query generation)

### Installation

```bash
# Clone and navigate to the project
cd ai-agent-project-tutorial

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e ".[dev]"
```

### Optional dependencies

```bash
# For local embeddings (no API key needed)
pip install -e ".[local]"

# For metadata extraction
pip install -e ".[extraction]"

# For evaluation plots
pip install -e ".[plot]"

# Install everything
pip install -e ".[dev,local,extraction,plot]"
```

### Environment variables

Create a `.env` file or export directly:

```bash
export OPENAI_API_KEY="sk-..."
```

### Configuration

All settings live in `config.yaml` and can be overridden via environment variables (prefix `RL_`) or CLI arguments:

```yaml
embeddings:
  backend: openai               # or "local"
  model: text-embedding-3-small

chunking:
  chunk_size: 300
  chunk_overlap: 50

training:
  total_timesteps: 12000
  top_k_candidates: 8
```

See [`config.yaml`](config.yaml) for all available options.

## Usage

### CLI

```bash
# Train a new RL retriever from a knowledge source
rl-retriever train --source https://example.com/docs --output ./model_output

# Train from a local file or directory
rl-retriever train --source ./my_docs/ --output ./model_output

# Query a trained model
rl-retriever query --model ./model_output --question "What is the battery life?"
```

### Python API

```python
from rl_memory_retrieval import Pipeline
from rl_memory_retrieval.config import Settings

# From config file
pipeline = Pipeline.from_config("config.yaml")
pipeline.train(source="https://example.com/docs", output_dir="./model")

# Query
result = pipeline.query("What is the battery life of Pulse?")
print(result["answer"])
```

## Testing

Run the full test suite:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run all tests with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_config.py -v
python -m pytest tests/test_chunker.py -v

# Run with coverage report
python -m pytest tests/ -v --cov=rl_memory_retrieval --cov-report=term-missing
```

### Test overview

| Test File | What it covers |
|-----------|---------------|
| `test_config.py` | Settings loading, defaults, overrides |
| `test_chunker.py` | Text splitting, overlap, edge cases |
| `test_features.py` | RL state feature engineering |
| `test_environment.py` | Gymnasium env reset/step/reward |
| `test_query_gen.py` | QA response parsing |
| `test_retriever.py` | Baseline retrieval logic |
| `test_integration.py` | End-to-end flow (no API calls) |

> **Note:** Tests are designed to run without API calls. They use mocks or synthetic data to verify logic without requiring an OpenAI API key.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RL Algorithm | PPO via [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) |
| RL Environment | [Gymnasium](https://gymnasium.farama.org/) |
| Embeddings | [OpenAI API](https://platform.openai.com/) / [sentence-transformers](https://sbert.net/) |
| Web Crawling | [Crawl4AI](https://github.com/unclecode/crawl4ai) |
| File Conversion | [markitdown](https://github.com/microsoft/markitdown) |
| Config | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) + YAML |
| CLI | [Click](https://click.palletsprojects.com/) |
| Testing | [pytest](https://docs.pytest.org/) |

## References

- Original tutorial: [Build a Reinforcement Learning Powered Agent for Long-Term Memory Retrieval](00_rl_agent_long_term_memory_retrieval.md)
- Design spec: [`docs/superpowers/specs/2026-04-28-rl-memory-retrieval-design.md`](docs/superpowers/specs/2026-04-28-rl-memory-retrieval-design.md)
