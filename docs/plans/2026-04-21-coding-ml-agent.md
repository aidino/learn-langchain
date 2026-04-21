# Coding ML Agent Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Build a DeepAgents-based coding ML agent with orchestrator + 2 subagents + AIO Sandbox to solve the Spaceship Titanic classification problem.

**Architecture:** Orchestrator agent (ML Project Manager) delegates to data-analyst and model-trainer subagents. All code execution happens inside AIO Sandbox container via custom backend. HITL checkpoints at 3 stages. 3 custom skills provide procedural knowledge.

**Tech Stack:** DeepAgents v0.5.3, LangChain, agent-sandbox SDK, z.ai (glm-5-turbo), scikit-learn, pandas, numpy

---

### Task 1: Project Scaffolding

**Files:**
- Create: `coding-ml-agent/.env`
- Create: `coding-ml-agent/deepagents.toml`

**Step 1: Create `.env` template**

```env
ZAI_API_KEY=your-api-key-here
```

**Step 2: Create `deepagents.toml`**

```toml
[agent]
name = "coding-ml-agent"
model = "openai:glm-5-turbo"
```

**Step 3: Commit**

```bash
git add coding-ml-agent/.env coding-ml-agent/deepagents.toml
git commit -m "feat(coding-ml-agent): scaffold project with env and config"
```

---

### Task 2: AIO Sandbox Custom Backend

**Files:**
- Create: `coding-ml-agent/sandbox_backend.py`

**Step 1: Create the custom backend**

Write `sandbox_backend.py` that wraps the `agent-sandbox` SDK into a DeepAgents-compatible backend by implementing the `BaseSandbox` interface.

```python
"""Custom AIO Sandbox backend for DeepAgents.

Wraps the agent-infra/sandbox SDK to provide Shell, File, and Jupyter
execution capabilities inside an isolated Docker container.
"""

from deepagents.backends.sandbox import BaseSandbox, SandboxResult
from agent_sandbox import Sandbox


class AIOSandboxBackend(BaseSandbox):
    """DeepAgents backend that delegates to AIO Sandbox container."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.client = Sandbox(base_url=base_url)
        ctx = self.client.sandbox.get_context()
        self.home_dir = ctx.home_dir

    def execute(self, command: str, **kwargs) -> SandboxResult:
        """Execute a shell command in the sandbox."""
        try:
            res = self.client.shell.exec_command(command=command)
            return SandboxResult(output=res.data.output, error=None)
        except Exception as e:
            return SandboxResult(output="", error=str(e))

    def read_file(self, filepath: str) -> str:
        """Read a file from the sandbox filesystem."""
        res = self.client.file.read_file(file=filepath)
        return res.data.content

    def write_file(self, filepath: str, content: str) -> None:
        """Write content to a file in the sandbox filesystem."""
        self.client.file.write_file(file=filepath, content=content)

    def ls(self, path: str) -> str:
        """List directory contents in the sandbox."""
        res = self.client.shell.exec_command(command=f"ls -la {path}")
        return res.data.output

    def upload_data(self, local_paths: dict[str, str]) -> None:
        """Upload local files to sandbox. local_paths = {sandbox_path: local_content}"""
        files = [(path, content.encode()) for path, content in local_paths.items()]
        self.client.file.write_file_batch(files=files)
```

**Step 2: Verify import structure**

```bash
cd coding-ml-agent && python -c "import sandbox_backend; print('OK')"
```

Expected: Module should be importable (may fail on missing deps — that's OK at this stage).

**Step 3: Commit**

```bash
git add coding-ml-agent/sandbox_backend.py
git commit -m "feat(coding-ml-agent): add AIO Sandbox custom backend"
```

---

### Task 3: Skill — EDA Workflow

**Files:**
- Create: `coding-ml-agent/skills/eda-workflow/SKILL.md`

**Step 1: Write the SKILL.md**

```markdown
---
name: eda-workflow
description: Use this skill when performing exploratory data analysis on a new tabular dataset. Follow these steps systematically to understand data shape, distributions, missing values, and feature relationships.
---

# EDA Workflow for Tabular Data

## Overview
Systematic exploratory data analysis procedure for tabular classification datasets. Produces a structured markdown report with key findings.

## Best Practices
- Always print results with `print()` so output is captured by the agent
- Process one analysis step at a time — do not generate all plots at once
- Save figures to files instead of showing inline when in sandbox environment
- Summarize findings in plain language, not just raw numbers

## Process

### Step 1: Data Overview
```python
import pandas as pd

df = pd.read_csv('/path/to/train.csv')
print(f"Shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic statistics:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nMissing percentage:\n{(df.isnull().sum() / len(df) * 100).round(2)}")
```

### Step 2: Target Distribution
```python
print(f"\nTarget distribution:\n{df['target_column'].value_counts()}")
print(f"\nTarget balance: {df['target_column'].value_counts(normalize=True).round(3)}")
```
Assess: Is the dataset balanced or imbalanced? If imbalanced (< 30/70), note this for modeling.

### Step 3: Feature Distributions
For each feature:
- **Numerical:** Print describe(), check for outliers (IQR method)
- **Categorical:** Print value_counts(), check cardinality

### Step 4: Feature-Target Relationships
```python
# Numerical features: group by target and compare means
print(df.groupby('target_column').mean(numeric_only=True))

# Categorical features: cross-tabulation with target
for col in categorical_columns:
    print(f"\n{col} vs Target:")
    print(pd.crosstab(df[col], df['target_column'], normalize='index').round(3))
```

### Step 5: Correlation Analysis
```python
import numpy as np
corr = df.select_dtypes(include=[np.number]).corr()
print(f"\nCorrelation with target:\n{corr['target_column'].sort_values(ascending=False)}")
```

### Step 6: Save EDA Report
Write findings to a markdown file summarizing:
- Dataset overview (shape, types)
- Missing value strategy recommendations
- Key feature insights
- Feature importance ranking
- Recommendations for feature engineering

## Common Pitfalls
- Do NOT load the entire dataset into a single print statement — use `.head()` or summaries
- Do NOT skip missing value analysis — it is critical for feature engineering decisions
- Do NOT create matplotlib plots without saving to file first (sandbox has no display)
```

**Step 2: Commit**

```bash
git add coding-ml-agent/skills/eda-workflow/SKILL.md
git commit -m "feat(coding-ml-agent): add EDA workflow skill"
```

---

### Task 4: Skill — Feature Engineering

**Files:**
- Create: `coding-ml-agent/skills/feature-engineering/SKILL.md`

**Step 1: Write the SKILL.md**

Write the skill with instructions for handling tabular data preprocessing:
- Missing value strategies (median for numeric, mode for categorical)
- Categorical encoding (LabelEncoder, OneHotEncoder)
- Domain-specific feature extraction (e.g., splitting Cabin into Deck/Num/Side)
- Aggregate feature creation (e.g., total spending)
- Data leakage prevention (fit only on train, transform both train and test)
- Output format: save `X_train.csv`, `y_train.csv`, `X_test.csv` with PassengerId preserved

**Anti-patterns to document:**
- Never fit scaler/encoder on test data
- Never drop PassengerId before saving (needed for submission)
- Always apply same transformations to train AND test

**Step 2: Commit**

```bash
git add coding-ml-agent/skills/feature-engineering/SKILL.md
git commit -m "feat(coding-ml-agent): add feature engineering skill"
```

---

### Task 5: Skill — Scikit-learn Modeling

**Files:**
- Create: `coding-ml-agent/skills/sklearn-modeling/SKILL.md`

**Step 1: Write the SKILL.md**

Write the skill with instructions for ML model training workflow:
- Baseline models: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- Evaluation: cross_val_score with 5-fold CV, metric = accuracy
- Comparison table format: Model | CV Mean | CV Std
- Hyperparameter tuning: GridSearchCV or RandomizedSearchCV for top model
- Submission generation: predict on test, format as PassengerId,Transported (True/False)
- Save submission to `/home/gem/output/submission.csv`

**Anti-patterns to document:**
- Never tune before establishing baselines
- Never use test labels for evaluation (we don't have them anyway)
- Never skip cross-validation — single train/test split is unreliable
- Always print the comparison table for orchestrator to report

**Step 2: Commit**

```bash
git add coding-ml-agent/skills/sklearn-modeling/SKILL.md
git commit -m "feat(coding-ml-agent): add sklearn modeling skill"
```

---

### Task 6: Subagent — data-analyst

**Files:**
- Create: `coding-ml-agent/subagents/data-analyst/AGENTS.md`
- Create: `coding-ml-agent/subagents/data-analyst/deepagents.toml`

**Step 1: Write AGENTS.md**

System prompt for data-analyst subagent:
- Role: Senior Data Scientist specialized in tabular data analysis
- Always execute code in sandbox using `execute` tool
- Always `print()` results so they are captured
- Load skills for EDA and feature engineering workflows
- Save reports to `/home/gem/reports/` and processed data to `/home/gem/data/`
- Keep final report to orchestrator ≤ 500 words — summarize, don't dump raw output
- Language: respond in the same language as the user's request

**Step 2: Write deepagents.toml**

```toml
[agent]
name = "data-analyst"
description = "Senior Data Scientist — performs EDA, data cleaning, feature engineering on tabular datasets. Executes Python code in sandbox using pandas and numpy. Returns structured summary reports."
```

**Step 3: Commit**

```bash
git add coding-ml-agent/subagents/data-analyst/
git commit -m "feat(coding-ml-agent): add data-analyst subagent"
```

---

### Task 7: Subagent — model-trainer

**Files:**
- Create: `coding-ml-agent/subagents/model-trainer/AGENTS.md`
- Create: `coding-ml-agent/subagents/model-trainer/deepagents.toml`

**Step 1: Write AGENTS.md**

System prompt for model-trainer subagent:
- Role: ML Engineer specialized in scikit-learn model training and evaluation
- Read processed data from `/home/gem/data/X_train.csv`, `y_train.csv`, `X_test.csv`
- Train multiple models, compare with cross-validation
- Tune best model's hyperparameters
- Generate submission.csv in correct Kaggle format
- Save model results report to `/home/gem/reports/model_results.md`
- Keep final report ≤ 500 words with metrics comparison table
- Language: respond in the same language as the user's request

**Step 2: Write deepagents.toml**

```toml
[agent]
name = "model-trainer"
description = "ML Engineer — trains, evaluates, and tunes scikit-learn classification models. Reads processed data, compares baselines via cross-validation, tunes hyperparameters, and generates Kaggle submission files."
```

**Step 3: Commit**

```bash
git add coding-ml-agent/subagents/model-trainer/
git commit -m "feat(coding-ml-agent): add model-trainer subagent"
```

---

### Task 8: Orchestrator AGENTS.md

**Files:**
- Create: `coding-ml-agent/AGENTS.md`

**Step 1: Write the orchestrator system prompt**

Write AGENTS.md with:
- Role: ML Project Manager / Orchestrator
- Never writes code directly — delegates to subagents via `task` tool
- Plans work using `write_todos` with 4-phase workflow
- After each phase, summarizes subagent results and asks user for approval via `ask_human`
- Knows about data location in sandbox: `/home/gem/data/`
- Knows about the Spaceship Titanic problem context
- Strict rule: Do NOT call `check_async_task` immediately after launching a subagent
- Language: match user's language

**Step 2: Commit**

```bash
git add coding-ml-agent/AGENTS.md
git commit -m "feat(coding-ml-agent): add orchestrator system prompt"
```

---

### Task 9: Main Script (main.py)

**Files:**
- Create: `coding-ml-agent/main.py`

**Step 1: Write main.py**

The entry point script that:
1. Loads `.env` with `python-dotenv`
2. Initializes LLM via `init_chat_model` with z.ai config
3. Creates `AIOSandboxBackend` instance
4. Uploads `train.csv` and `test.csv` to sandbox `/home/gem/data/`
5. Installs pandas, numpy, scikit-learn in sandbox via `execute("pip install ...")`
6. Defines subagent configs (data-analyst, model-trainer) with skills paths
7. Creates orchestrator agent via `create_deep_agent()` with:
   - model, backend, system_prompt (from AGENTS.md)
   - subagents list
   - skills paths
   - `interrupt_on={"ask_human": True}`
   - `MemorySaver` checkpointer for HITL
8. Runs agent invoke loop with user input
9. Handles HITL interrupts — display results, get user approval, resume
10. Downloads `submission.csv` from sandbox at the end

**Step 2: Test basic execution**

```bash
# Requires: Docker container running, .env configured
cd coding-ml-agent && python main.py
```

**Step 3: Commit**

```bash
git add coding-ml-agent/main.py
git commit -m "feat(coding-ml-agent): add main entry point with agent loop"
```

---

### Task 10: Integration Testing & Polish

**Step 1: Start AIO Sandbox**

```bash
docker run --security-opt seccomp=unconfined --rm -it -p 8080:8080 ghcr.io/agent-infra/sandbox:latest
```

**Step 2: Verify sandbox connectivity**

```bash
cd coding-ml-agent && python -c "
from agent_sandbox import Sandbox
c = Sandbox(base_url='http://localhost:8080')
print(c.sandbox.get_context().home_dir)
print(c.shell.exec_command(command='python3 --version').data.output)
"
```

**Step 3: Run full pipeline**

```bash
cd coding-ml-agent && python main.py
```

Test the full workflow:
- Say: "Giải bài Spaceship Titanic"
- Verify: Orchestrator creates plan → delegates EDA to data-analyst → pauses for HITL
- Approve EDA → verify feature engineering proceeds → pauses for HITL
- Approve FE → verify model training → pauses for HITL
- Approve results → verify submission.csv is generated and downloaded

**Step 4: Verify submission format**

```bash
head -5 coding-ml-agent/spaceship-titanic/submission.csv
# Expected:
# PassengerId,Transported
# 0013_01,True
# 0018_01,False
# ...
```

**Step 5: Final commit**

```bash
git add -A coding-ml-agent/
git commit -m "feat(coding-ml-agent): complete ML agent with full pipeline"
```
