"""Coding ML Agent — Main Entry Point.

Orchestrates a DeepAgents-based ML pipeline to solve the Spaceship Titanic
classification problem using an Orchestrator + 2 Subagents architecture.

Usage:
    1. Start AIO Sandbox:
       docker run --security-opt seccomp=unconfined --rm -it -p 8080:8080 \
           ghcr.io/agent-infra/sandbox:latest

    2. Configure .env:
       ZAI_API_KEY=your-api-key-here

    3. Run:
       cd coding-ml-agent && python main.py

Architecture:
    main.py → Orchestrator Agent (ML Project Manager)
                ├── data-analyst subagent (EDA + Feature Engineering)
                └── model-trainer subagent (Training + Submission)
              All code execution → AIO Sandbox (Docker container)

HITL Checkpoints:
    1. After EDA (Phase 1)
    2. After Feature Engineering (Phase 2)
    3. After Model Training (Phase 3)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent
from sandbox_backend import AIOSandboxBackend


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load environment variables
load_dotenv()

# Paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "spaceship-titanic"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Sandbox data directory
SANDBOX_DATA_DIR = "/home/gem/data"

# LLM config (z.ai / glm-5-turbo)
LLM_MODEL = os.getenv("LLM_MODEL", "openai:glm-5-turbo")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
LLM_API_KEY = os.getenv("ZAI_API_KEY")

# Sandbox config
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080")


# ---------------------------------------------------------------------------
# Load AGENTS.md system prompts
# ---------------------------------------------------------------------------

def load_system_prompt(filepath: Path) -> str:
    """Load a system prompt from an AGENTS.md file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Subagent Definitions
# ---------------------------------------------------------------------------

def build_subagents() -> list[dict]:
    """Build subagent configurations for data-analyst and model-trainer."""

    data_analyst_prompt = load_system_prompt(
        PROJECT_DIR / "subagents" / "data-analyst" / "AGENTS.md"
    )
    model_trainer_prompt = load_system_prompt(
        PROJECT_DIR / "subagents" / "model-trainer" / "AGENTS.md"
    )

    return [
        {
            "name": "data-analyst",
            "description": (
                "Senior Data Scientist — performs EDA, data cleaning, "
                "feature engineering on tabular datasets. Executes Python "
                "code in sandbox using pandas and numpy. Returns structured "
                "summary reports."
            ),
            "system_prompt": data_analyst_prompt,
            "skills": [
                str(PROJECT_DIR / "skills" / "eda-workflow"),
                str(PROJECT_DIR / "skills" / "feature-engineering"),
            ],
        },
        {
            "name": "model-trainer",
            "description": (
                "ML Engineer — trains, evaluates, and tunes scikit-learn "
                "classification models. Reads processed data, compares "
                "baselines via cross-validation, tunes hyperparameters, "
                "and generates Kaggle submission files."
            ),
            "system_prompt": model_trainer_prompt,
            "skills": [
                str(PROJECT_DIR / "skills" / "sklearn-modeling"),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Sandbox Setup
# ---------------------------------------------------------------------------

def setup_sandbox(backend: AIOSandboxBackend) -> None:
    """Upload data files and install Python packages in the sandbox."""

    print("📦 Setting up sandbox environment...")

    # Create directories
    backend.execute(f"mkdir -p {SANDBOX_DATA_DIR} /home/gem/reports /home/gem/output")

    # Upload data files
    print(f"  📄 Uploading {TRAIN_CSV.name}...")
    backend.upload(str(TRAIN_CSV), f"{SANDBOX_DATA_DIR}/train.csv")

    print(f"  📄 Uploading {TEST_CSV.name}...")
    backend.upload(str(TEST_CSV), f"{SANDBOX_DATA_DIR}/test.csv")

    # Install Python packages
    print("  📦 Installing pandas, numpy, scikit-learn...")
    result = backend.setup_environment(["pandas", "numpy", "scikit-learn"])
    print(f"  ✅ Package install complete")

    # Verify data is accessible
    ls_result = backend.execute(f"ls -la {SANDBOX_DATA_DIR}")
    print(f"  📂 Sandbox data directory:\n{ls_result.output}")


# ---------------------------------------------------------------------------
# HITL Agent Loop
# ---------------------------------------------------------------------------

def run_agent_loop(agent, config: dict) -> None:
    """Run the agent with HITL (Human-in-the-Loop) interrupt handling.

    The agent will pause at `ask_human` tool calls, display results to
    the user, and resume after receiving user feedback.
    """

    print("\n" + "=" * 60)
    print("🤖 Coding ML Agent — Spaceship Titanic")
    print("=" * 60)

    # Get initial user input
    user_input = input("\n💬 Enter your request (or 'quit' to exit): ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("👋 Goodbye!")
        return

    # Initial invocation
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    while True:
        # Check if agent interrupted (HITL checkpoint)
        if result.get("__interrupt__"):
            interrupt_info = result["__interrupt__"]
            print("\n" + "-" * 60)
            print("⏸️  HITL CHECKPOINT — Agent is waiting for your input")
            print("-" * 60)

            # Display the interrupt details
            if isinstance(interrupt_info, list):
                for item in interrupt_info:
                    if hasattr(item, "value"):
                        print(f"\n📋 Agent says:\n{item.value}")
                    else:
                        print(f"\n📋 {item}")
            else:
                print(f"\n📋 Agent says:\n{interrupt_info}")

            # Get user response
            user_response = input("\n💬 Your response (or 'quit'): ").strip()
            if user_response.lower() in ("quit", "exit", "q"):
                print("👋 Stopping agent.")
                break

            # Resume agent with user response
            from langchain_core.messages import HumanMessage
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_response)]},
                config=config,
            )

        else:
            # Agent completed without interruption
            last_message = result["messages"][-1]
            print("\n" + "=" * 60)
            print("✅ Agent completed!")
            print("=" * 60)
            if hasattr(last_message, "content"):
                print(f"\n{last_message.content}")
            break


# ---------------------------------------------------------------------------
# Download Results
# ---------------------------------------------------------------------------

def download_results(backend: AIOSandboxBackend) -> None:
    """Download submission.csv and reports from sandbox to local filesystem."""

    output_dir = PROJECT_DIR / "spaceship-titanic"

    # Download submission
    submission_path = "/home/gem/output/submission.csv"
    local_submission = str(output_dir / "submission.csv")
    try:
        backend.download(submission_path, local_submission)
        print(f"\n📥 Downloaded submission: {local_submission}")

        # Preview submission
        with open(local_submission, "r") as f:
            lines = f.readlines()
            print(f"  Rows: {len(lines) - 1}")
            print(f"  Preview:")
            for line in lines[:5]:
                print(f"    {line.strip()}")
    except Exception as e:
        print(f"\n⚠️  Could not download submission: {e}")

    # Download reports
    reports_dir = PROJECT_DIR / "spaceship-titanic" / "reports"
    reports_dir.mkdir(exist_ok=True)

    for report_name in ["eda_report.md", "model_results.md"]:
        try:
            backend.download(
                f"/home/gem/reports/{report_name}",
                str(reports_dir / report_name),
            )
            print(f"📥 Downloaded report: {reports_dir / report_name}")
        except Exception:
            pass  # Report may not exist yet


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point for the Coding ML Agent."""

    # Validate environment
    if not LLM_API_KEY:
        print("❌ Error: ZAI_API_KEY not set in .env file")
        print("  Create a .env file with: ZAI_API_KEY=your-key-here")
        sys.exit(1)

    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print(f"❌ Error: Data files not found in {DATA_DIR}")
        print(f"  Expected: {TRAIN_CSV} and {TEST_CSV}")
        sys.exit(1)

    # Initialize LLM
    print("🔧 Initializing LLM...")
    model = init_chat_model(
        LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )

    # Initialize sandbox backend
    print("🔧 Connecting to AIO Sandbox...")
    try:
        backend = AIOSandboxBackend(base_url=SANDBOX_URL)
        print(f"  ✅ Connected (home: {backend.home_dir})")
    except Exception as e:
        print(f"❌ Error: Could not connect to sandbox at {SANDBOX_URL}")
        print(f"  Start it with: docker run --security-opt seccomp=unconfined "
              f"--rm -it -p 8080:8080 ghcr.io/agent-infra/sandbox:latest")
        print(f"  Error: {e}")
        sys.exit(1)

    # Setup sandbox (upload data, install packages)
    setup_sandbox(backend)

    # Load orchestrator system prompt
    orchestrator_prompt = load_system_prompt(PROJECT_DIR / "AGENTS.md")

    # Build subagent configs
    subagents = build_subagents()

    # Create checkpointer for HITL
    checkpointer = MemorySaver()

    # Create orchestrator agent
    print("\n🤖 Creating orchestrator agent...")
    agent = create_deep_agent(
        model=model,
        system_prompt=orchestrator_prompt,
        memory=[str(PROJECT_DIR / "AGENTS.md")],
        skills=[str(PROJECT_DIR / "skills")],
        subagents=subagents,
        backend=backend,
        interrupt_on={"ask_human": True},
        checkpointer=checkpointer,
    )
    print("  ✅ Agent ready")

    # Run agent with HITL loop
    config = {"configurable": {"thread_id": "spaceship-titanic-session"}}

    try:
        run_agent_loop(agent, config)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user.")

    # Download results from sandbox
    print("\n📥 Downloading results from sandbox...")
    download_results(backend)

    print("\n🏁 Done!")


if __name__ == "__main__":
    main()
