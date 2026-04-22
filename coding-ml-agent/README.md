# 🤖 Coding ML Agent

A **DeepAgents v0.5.3** coding agent that solves the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) classification problem using an Orchestrator + Subagents architecture with Human-in-the-Loop (HITL) checkpoints.

The agent **plans, delegates, and executes** a full ML pipeline — from EDA to submission — entirely inside an isolated Docker sandbox.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                  Python Script (main.py)                │
│  Load .env → Init LLM → Start sandbox → Agent loop     │
└────────────────────────┬───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│             ORCHESTRATOR AGENT                          │
│  Role: ML Project Manager (never writes code)           │
│  Model: glm-5-turbo (z.ai)                              │
│  HITL: interrupt at 3 checkpoints                       │
│                                                         │
│  ┌─────────────────┐    ┌──────────────────┐           │
│  │  data-analyst   │    │  model-trainer   │           │
│  │  (subagent)     │    │  (subagent)      │           │
│  │  Skills:        │    │  Skills:         │           │
│  │  • eda-workflow │    │  • sklearn-      │           │
│  │  • feature-eng  │    │    modeling      │           │
│  └────────┬────────┘    └────────┬─────────┘           │
└───────────┼──────────────────────┼─────────────────────┘
            │                      │
┌───────────▼──────────────────────▼─────────────────────┐
│          AIO SANDBOX (Docker Container)                  │
│  Shell API  │  File API  │  Jupyter API                  │
│  Pre-installed: pandas, numpy, scikit-learn              │
└────────────────────────────────────────────────────────┘
```

## HITL Workflow

The agent pauses at **3 checkpoints** to present results and ask for approval:

| Phase | What Happens | Checkpoint |
|---|---|---|
| **Phase 1** | EDA — data overview, distributions, correlations | ⏸️ Review EDA findings |
| **Phase 2** | Feature Engineering — missing values, encoding, new features | ⏸️ Review feature strategy |
| **Phase 3** | Model Training — baselines, comparison, hyperparameter tuning | ⏸️ Review model results |
| **Phase 4** | Submission — predict test set, generate `submission.csv` | ✅ Done |

## Project Structure

```
coding-ml-agent/
├── main.py                          # Entry point + HITL agent loop
├── sandbox_backend.py               # AIO Sandbox custom backend (BaseSandbox)
├── AGENTS.md                        # Orchestrator system prompt
├── deepagents.toml                  # Agent config
├── .env                             # API keys (not committed)
├── .env.example                     # API key template
├── skills/
│   ├── eda-workflow/SKILL.md        # EDA procedure
│   ├── feature-engineering/SKILL.md # Feature engineering procedure
│   └── sklearn-modeling/SKILL.md    # Model training procedure
├── subagents/
│   ├── data-analyst/
│   │   ├── AGENTS.md                # Data scientist system prompt
│   │   └── deepagents.toml          # Subagent config
│   └── model-trainer/
│       ├── AGENTS.md                # ML engineer system prompt
│       └── deepagents.toml          # Subagent config
└── spaceship-titanic/
    ├── overview.md                  # Problem description
    ├── train.csv                    # Training data (8693 rows)
    ├── test.csv                     # Test data
    └── sample_submission.csv        # Expected submission format
```

## Prerequisites

- **Python 3.10+**
- **Docker** (for AIO Sandbox)
- **z.ai API key** (or any OpenAI-compatible endpoint)

## Setup

### 1. Create virtual environment

```bash
cd coding-ml-agent
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install deepagents langchain langgraph agent-sandbox python-dotenv
```

### 3. Configure API key

```bash
cp .env.example .env
# Edit .env and set your API key:
# ZAI_API_KEY=your-actual-key
```

You can also change the LLM model and endpoint by setting environment variables:

| Variable | Default | Description |
|---|---|---|
| `ZAI_API_KEY` | *(required)* | API key for the LLM provider |
| `LLM_MODEL` | `openai:glm-5-turbo` | LangChain model identifier |
| `LLM_BASE_URL` | `https://api.z.ai/api/coding/paas/v4` | OpenAI-compatible API endpoint |
| `SANDBOX_URL` | `http://localhost:8080` | AIO Sandbox URL |

### 4. Start AIO Sandbox

```bash
docker run --security-opt seccomp=unconfined --rm -it -p 8080:8080 \
    ghcr.io/agent-infra/sandbox:latest
```

Wait for the `supervisord started` message before proceeding.

## Usage

### Run the agent

```bash
cd coding-ml-agent
source .venv/bin/activate
python main.py
```

The agent will:
1. Upload `train.csv` and `test.csv` to the sandbox
2. Install pandas, numpy, scikit-learn in the sandbox
3. Prompt you for input

### Example interaction

```
💬 Enter your request: Giải bài Spaceship Titanic

🤖 Orchestrator plans 4-phase workflow...
   → Delegates EDA to data-analyst subagent
   → data-analyst runs code in sandbox, produces EDA report

⏸️  HITL CHECKPOINT
📋 Phase 1 results: Dataset has 8693 rows, 14 columns...
   Target is balanced (50.4% / 49.6%)...

💬 Your response: Looks good, proceed

   → Delegates feature engineering to data-analyst
   → Produces X_train.csv, y_train.csv, X_test.csv

⏸️  HITL CHECKPOINT
📋 Phase 2 results: Created 15 features, handled missing values...

💬 Your response: OK

   → Delegates model training to model-trainer
   → Trains LogisticRegression, RandomForest, GradientBoosting
   → Tunes best model with RandomizedSearchCV

⏸️  HITL CHECKPOINT
📋 Phase 3 results:
   | Model              | CV Mean | CV Std |
   | GradientBoosting   | 0.8012  | 0.0098 |
   | RandomForest       | 0.7934  | 0.0112 |

💬 Your response: Generate submission

✅ Agent completed!
📥 Downloaded submission: spaceship-titanic/submission.csv
```

## Testing

### Verify sandbox connectivity

```bash
source .venv/bin/activate
python3 -c "
from agent_sandbox import Sandbox
c = Sandbox(base_url='http://localhost:8080')
print('Home dir:', c.sandbox.get_context().home_dir)
print('Python:', c.shell.exec_command(command='python3 --version').data.output)
"
```

Expected output:
```
Home dir: /home/gem
Python: Python 3.10.12
```

### Verify backend integration

```bash
python3 -c "
from sandbox_backend import AIOSandboxBackend

backend = AIOSandboxBackend(base_url='http://localhost:8080')
print(f'ID: {backend.id}')
print(f'Home: {backend.home_dir}')

# Test execute
result = backend.execute('echo Hello from sandbox')
print(f'Execute: {result.output.strip()}')

# Test upload + download
backend.upload_files([('/tmp/test.txt', b'hello world')])
resp = backend.download_files(['/tmp/test.txt'])
print(f'Round-trip: {resp[0].content}')
"
```

### Verify all imports

```bash
python3 -c "
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from sandbox_backend import AIOSandboxBackend
print('All imports OK')
"
```

## Key Concepts Applied

| DeepAgents Concept | How It's Used |
|---|---|
| **Subagents** | 2 specialized subagents (data-analyst, model-trainer) |
| **Context Isolation** | Each subagent has its own context, returns ≤ 500-word summary |
| **Skills** | 3 custom skills loaded on-demand (EDA, FE, Modeling) |
| **HITL** | `interrupt_on={"ask_human": True}` at 3 checkpoints |
| **Sandbox** | Custom `AIOSandboxBackend` extending `BaseSandbox` |
| **Context Engineering** | Results offloaded to files, reports size-limited |
| **Memory** | `AGENTS.md` loaded as system prompt via middleware |
| **Planning** | `write_todos` for 4-phase workflow |

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: agent_sandbox` | Run `pip install agent-sandbox` in the venv |
| `ConnectionError` to sandbox | Ensure Docker container is running on port 8080 |
| `ZAI_API_KEY not set` | Copy `.env.example` to `.env` and add your key |
| Sandbox command timeout | Check Docker has enough resources allocated |
| `TypeError: Can't instantiate abstract class` | You're using an older `sandbox_backend.py` — pull latest |

## License

Part of the [learn-langchain](../) project.
