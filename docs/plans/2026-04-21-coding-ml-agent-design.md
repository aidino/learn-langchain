# Coding ML Agent — Design Document

> **Date:** 2026-04-21  
> **Goal:** Xây dựng coding ML agent sử dụng DeepAgents v0.5.3 để giải bài toán Spaceship Titanic  
> **Status:** Approved

## 1. Tổng quan

Xây dựng một **Coding ML Agent** sử dụng framework DeepAgents v0.5.3, ứng dụng toàn bộ kiến thức đã học trong khóa (subagents, skills, HITL, sandbox, context engineering). Agent giải quyết bài toán Spaceship Titanic — binary classification dự đoán hành khách bị transported hay không, sử dụng scikit-learn, pandas, numpy.

### Quyết định thiết kế chính

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Kiến trúc | Orchestrator + 2 Subagents | Ứng dụng context isolation, tương tự research-agent |
| HITL | Interrupt tại 3 mốc (EDA, FE, Training) | Kiểm soát chất lượng từng giai đoạn |
| LLM | z.ai (glm-5-turbo), OpenAI-compatible | Model do user chọn |
| Sandbox | AIO Sandbox (agent-infra/sandbox) | Docker container, chạy code cách ly |
| Interaction | Python script (main.py) | Linh hoạt, tùy chỉnh dễ |
| Skills | 3 custom skills (EDA, FE, Modeling) | Tailored cho bài toán tabular ML |

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Python Script (main.py)                │
│  - Load .env, init LLM, start sandbox, run agent loop    │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│              ORCHESTRATOR AGENT                           │
│  Model: glm-5-turbo (z.ai)                               │
│  Role: ML Project Manager                                │
│  HITL: interrupt tại 3 mốc                                │
│                                                           │
│  Tools: write_todos (planning)                            │
│  Memory: AGENTS.md                                        │
│                                                           │
│  ┌─────────────────┐    ┌──────────────────┐             │
│  │  data-analyst   │    │  model-trainer   │             │
│  │  (subagent)     │    │  (subagent)      │             │
│  └────────┬────────┘    └────────┬─────────┘             │
└───────────┼──────────────────────┼───────────────────────┘
            │                      │
┌───────────▼──────────────────────▼───────────────────────┐
│           AIO SANDBOX (Docker Container)                  │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐                  │
│  │  Shell   │ │  Jupyter │ │   File   │                  │
│  │  API     │ │  API     │ │   API    │                  │
│  └─────────┘ └──────────┘ └──────────┘                  │
│  Pre-installed: pandas, numpy, scikit-learn               │
│  Data: train.csv, test.csv (uploaded at init)             │
└──────────────────────────────────────────────────────────┘
```

## 3. Orchestrator Agent

### System Prompt (AGENTS.md)
- Role: ML Project Manager — không tự viết code, chỉ điều phối
- Lập kế hoạch bằng `write_todos`
- Giao việc cho subagents bằng `task`
- Tổng hợp kết quả và trình bày cho user
- Dừng lại tại các mốc HITL để xin phê duyệt

### HITL Implementation
Sử dụng `interrupt_on` trên tool `ask_human`:
- Subagent hoàn thành → trả kết quả về orchestrator
- Orchestrator tóm tắt → gọi `ask_human` để hỏi ý kiến user
- User approve → orchestrator tiếp tục giai đoạn tiếp

### Workflow mẫu (write_todos)
```
Phase 1: Exploratory Data Analysis
  - [ ] Giao data-analyst: phân tích tổng quan data
  - [ ] Giao data-analyst: phân tích phân phối target & features
  - [ ] Giao data-analyst: phân tích tương quan
  → CHECKPOINT: Trình bày EDA cho user

Phase 2: Feature Engineering  
  - [ ] Giao data-analyst: xử lý missing values
  - [ ] Giao data-analyst: encode categorical, tạo features mới
  - [ ] Giao data-analyst: lưu processed data ra file
  → CHECKPOINT: Trình bày feature strategy cho user

Phase 3: Model Training & Evaluation
  - [ ] Giao model-trainer: train baseline models
  - [ ] Giao model-trainer: so sánh metrics, chọn best model  
  - [ ] Giao model-trainer: tune hyperparameters
  → CHECKPOINT: Trình bày results cho user

Phase 4: Submission
  - [ ] Giao model-trainer: predict test set & tạo submission.csv
```

## 4. Subagents

### 4.1 data-analyst
- **Skills:** `eda-workflow`, `feature-engineering`
- **Role:** Data scientist chuyên phân tích và xử lý dữ liệu tabular
- Chạy code trong sandbox qua tool `execute`
- Luôn `print()` kết quả
- Lưu kết quả phân tích thành file markdown trong sandbox
- Lưu processed data ra CSV cho model-trainer
- Giới hạn báo cáo ≤ 500 từ

### 4.2 model-trainer
- **Skills:** `sklearn-modeling`
- **Role:** ML Engineer chuyên training & evaluation
- Đọc processed data từ CSV đã tạo bởi data-analyst
- Train nhiều models, so sánh bằng cross-validation
- Tune hyperparameter cho best model
- Generate `submission.csv` theo đúng format Kaggle
- Giới hạn báo cáo ≤ 500 từ kèm bảng metrics

## 5. Custom Skills

### 5.1 `eda-workflow`
**Trigger:** Khi data-analyst nhận task khám phá dữ liệu mới  
**Instructions:**
1. `df.info()`, `df.describe()`, `df.isnull().sum()` — tổng quan
2. Phân tích target distribution (balance/imbalance)
3. Phân phối từng feature (numerical → histogram, categorical → value_counts)
4. Correlation matrix + feature-target relationship
5. Lưu kết quả EDA thành file markdown report
- **Anti-patterns:** Không tạo quá nhiều plot cùng lúc, luôn `print()` kết quả

### 5.2 `feature-engineering`
**Trigger:** Khi data-analyst xử lý & tạo features cho tabular data  
**Instructions:**
1. Missing values: median cho numeric, mode cho categorical
2. Encoding: Label/OneHot cho categorical, xử lý Cabin (Deck/Num/Side)
3. Feature creation: aggregate features (total spending), group-based features
4. Scaling: StandardScaler/MinMaxScaler khi cần
5. Output: Lưu `X_train.csv`, `X_test.csv`, `y_train.csv`
- **Anti-patterns:** Không fit scaler trên test set (data leakage)

### 5.3 `sklearn-modeling`
**Trigger:** Khi model-trainer cần train, evaluate, tune models  
**Instructions:**
1. Baseline: LogisticRegression → RandomForest → GradientBoosting
2. Evaluation: `cross_val_score` (5-fold CV)
3. Comparison: Bảng so sánh accuracy từng model
4. Tuning: `GridSearchCV`/`RandomizedSearchCV` cho best model
5. Submission: `model.predict(X_test)` → format `PassengerId,Transported` → `submission.csv`
- **Anti-patterns:** Không tune trước khi có baseline comparison

## 6. Sandbox Integration

### AIO Sandbox (agent-infra/sandbox)
- Docker: `docker run --security-opt seccomp=unconfined --rm -it -p 8080:8080 ghcr.io/agent-infra/sandbox:latest`
- SDK: `pip install agent-sandbox`
- Custom Backend: `AIOSandboxBackend` kế thừa `BaseSandbox`, wrap SDK calls
- APIs used: `shell.exec_command`, `file.read_file`, `file.write_file`, `jupyter.execute_code`

### Data Flow
```
INIT: upload train.csv, test.csv → /home/gem/data/
  ↓
PHASE 1 (EDA): data-analyst → /home/gem/reports/eda_report.md → ⏸️ HITL
  ↓
PHASE 2 (FE): data-analyst → /home/gem/data/X_train.csv, y_train.csv, X_test.csv → ⏸️ HITL
  ↓
PHASE 3 (Train): model-trainer → /home/gem/reports/model_results.md + /home/gem/output/submission.csv → ⏸️ HITL
  ↓
PHASE 4: download submission.csv → local
```

## 7. LLM Configuration

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai:glm-5-turbo",
    base_url="https://api.z.ai/api/coding/paas/v4",
    api_key=os.getenv("ZAI_API_KEY"),
)
```

## 8. File Structure

```
coding-ml-agent/
├── main.py                      ← Entry point + agent loop
├── sandbox_backend.py           ← AIO Sandbox custom backend
├── .env                         ← ZAI_API_KEY
├── AGENTS.md                    ← Orchestrator system prompt
├── deepagents.toml              ← Agent config
├── skills/
│   ├── eda-workflow/
│   │   └── SKILL.md
│   ├── feature-engineering/
│   │   └── SKILL.md
│   └── sklearn-modeling/
│       └── SKILL.md
├── subagents/
│   ├── data-analyst/
│   │   ├── AGENTS.md
│   │   └── deepagents.toml
│   └── model-trainer/
│       ├── AGENTS.md
│       └── deepagents.toml
└── spaceship-titanic/
    ├── overview.md
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## 9. DeepAgents Concepts Ứng Dụng

| Concept | Cách ứng dụng |
|---|---|
| Subagents | 2 subagents chuyên biệt (data-analyst, model-trainer) |
| Context Isolation | Mỗi subagent có context riêng, chỉ trả summary |
| Skills | 3 custom skills (EDA, FE, Modeling) nạp on-demand |
| HITL | interrupt_on ask_human tại 3 checkpoint |
| Sandbox | AIO Sandbox custom backend wrap BaseSandbox |
| Context Engineering | Offloading results to files, giới hạn report ≤ 500 từ |
| Memory | AGENTS.md cho orchestrator + subagent prompts |
| Planning | write_todos cho 4-phase workflow |
| Backend | Custom AIOSandboxBackend kế thừa BaseSandbox |
