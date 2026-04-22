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
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from deepagents import create_deep_agent
from sandbox_backend import AIOSandboxBackend


# ---------------------------------------------------------------------------
# Rich Console Setup
# ---------------------------------------------------------------------------

THEME = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "thinking": "dim italic",
    "output": "green",
    "node": "bold magenta",
    "tool": "bold cyan",
    "llm": "bold yellow",
    "hitl": "bold white on blue",
})

console = Console(theme=THEME)


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
# Rich Progress Callback Handler
# ---------------------------------------------------------------------------

class AgentProgressHandler(BaseCallbackHandler):
    """Rich-formatted real-time logging of LLM calls and tool usage.

    Distinguishes between:
    - 🧠 Thinking: LLM reasoning / intermediate steps (dim italic)
    - 📤 Output: Final responses and results (bright green)
    - 🔧 Tools: Tool invocations with syntax-highlighted input/output
    """

    def __init__(self):
        self.llm_call_count = 0
        self.tool_call_count = 0
        self._t0 = None
        self._spinner = None

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.llm_call_count += 1
        self._t0 = time.time()
        name = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("id", ["?"])[-1]
        )
        n = sum(len(b) for b in messages) if messages else 0
        console.print()
        console.print(
            f"  [llm]🧠 LLM #{self.llm_call_count}[/llm]  "
            f"[dim]{n} messages → {name}[/dim]"
        )

    def on_llm_end(self, response, **kwargs):
        dt = time.time() - self._t0 if self._t0 else 0
        tok = ""
        if hasattr(response, "llm_output") and response.llm_output:
            u = response.llm_output.get("token_usage", {})
            if u:
                tok = (
                    f"  [dim]tokens: "
                    f"{u.get('prompt_tokens', '?')} in / "
                    f"{u.get('completion_tokens', '?')} out[/dim]"
                )
        console.print(f"  [success]✓ Response[/success] [dim]({dt:.1f}s)[/dim]{tok}")

    def on_llm_error(self, error, **kwargs):
        dt = time.time() - self._t0 if self._t0 else 0
        console.print(f"  [error]✗ LLM error ({dt:.1f}s): {error}[/error]")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_call_count += 1
        tool_name = serialized.get("name", "?")
        console.print(f"  [tool]🔧 Tool #{self.tool_call_count}: {tool_name}[/tool]")

        # Show tool input (truncated) with syntax highlighting if it looks like code
        s = str(input_str).strip()
        if s and len(s) > 10:
            if len(s) > 500:
                s = s[:500] + "…"
            if "\n" in s or "import " in s or "def " in s:
                try:
                    console.print(Syntax(s, "python", theme="monokai", line_numbers=False, padding=1))
                except Exception:
                    console.print(f"     [dim]{s}[/dim]")
            else:
                console.print(f"     [dim]{s}[/dim]")

    def on_tool_end(self, output, **kwargs):
        s = str(output).strip()
        if not s:
            console.print("     [dim](empty output)[/dim]")
            return
        if len(s) > 300:
            s = s[:300] + "…"
        console.print(Panel(
            Text(s, style="dim"),
            title="[tool]Tool Output[/tool]",
            border_style="cyan",
            padding=(0, 1),
            expand=False,
        ))

    def on_tool_error(self, error, **kwargs):
        console.print(Panel(
            str(error),
            title="[error]Tool Error[/error]",
            border_style="red",
            expand=False,
        ))


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

    console.print("\n[info]📦 Setting up sandbox environment...[/info]")

    # Create directories
    backend.execute(f"mkdir -p {SANDBOX_DATA_DIR} /home/gem/reports /home/gem/output")

    # Upload data files
    console.print(f"  📄 Uploading [bold]{TRAIN_CSV.name}[/bold]...")
    backend.upload(str(TRAIN_CSV), f"{SANDBOX_DATA_DIR}/train.csv")

    console.print(f"  📄 Uploading [bold]{TEST_CSV.name}[/bold]...")
    backend.upload(str(TEST_CSV), f"{SANDBOX_DATA_DIR}/test.csv")

    # Install Python packages
    console.print("  📦 Installing [bold]pandas, numpy, scikit-learn[/bold]...")
    result = backend.setup_environment(["pandas", "numpy", "scikit-learn"])
    console.print("  [success]✓ Package install complete[/success]")

    # Verify data is accessible
    ls_result = backend.execute(f"ls -la {SANDBOX_DATA_DIR}")
    console.print(Panel(
        ls_result.output.strip(),
        title="[info]📂 Sandbox Data Directory[/info]",
        border_style="cyan",
        padding=(0, 1),
    ))


# ---------------------------------------------------------------------------
# HITL Agent Loop
# ---------------------------------------------------------------------------

def _classify_message(msg):
    """Classify a message as 'thinking', 'tool_call', or 'output'.

    Returns (category, content) tuple.
    """
    msg_type = getattr(msg, "type", "")
    content = getattr(msg, "content", "")

    # Tool calls are their own category
    if msg_type == "tool":
        return "tool_result", content

    # AI messages with tool_calls are thinking (deciding what to do)
    if msg_type == "ai":
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            return "thinking", content

        # If there's additional_kwargs with thinking/reasoning
        additional = getattr(msg, "additional_kwargs", {})
        if additional.get("reasoning_content") or additional.get("thinking"):
            thinking = additional.get("reasoning_content") or additional.get("thinking", "")
            return "thinking", thinking

        # Final AI messages without tool calls are output
        if content:
            return "output", content

    return "other", content


def _stream_agent(agent, input_data, config):
    """Stream agent execution with Rich formatting.

    Returns (interrupted, interrupt_info, last_messages).
    """
    interrupted = False
    interrupt_info = None
    last_messages = []

    for event in agent.stream(input_data, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__interrupt__":
                interrupted = True
                interrupt_info = node_output
                continue

            # Show node transition
            console.print()
            console.rule(f"[node]📍 {node_name}[/node]", style="magenta")

            if isinstance(node_output, dict) and "messages" in node_output:
                raw = node_output["messages"]
                # Handle LangGraph Overwrite objects
                if hasattr(raw, "value"):
                    raw = raw.value
                if not isinstance(raw, (list, tuple)):
                    raw = [raw] if raw else []
                last_messages = raw
                for msg in last_messages:
                    category, content = _classify_message(msg)

                    if not content:
                        continue

                    if category == "thinking":
                        # Thinking = dim, italic, collapsible feel
                        txt = content
                        if len(txt) > 800:
                            txt = txt[:800] + " …"
                        console.print(Panel(
                            Text(txt, style="thinking"),
                            title="[dim]💭 Thinking[/dim]",
                            border_style="dim",
                            padding=(0, 1),
                            expand=True,
                        ))

                    elif category == "output":
                        # Output = bright, prominent
                        try:
                            console.print(Panel(
                                Markdown(content),
                                title="[success]📤 Output[/success]",
                                border_style="green",
                                padding=(1, 2),
                                expand=True,
                            ))
                        except Exception:
                            console.print(Panel(
                                content,
                                title="[success]📤 Output[/success]",
                                border_style="green",
                                padding=(1, 2),
                            ))

                    elif category == "tool_result":
                        # Tool results — compact
                        txt = content
                        if len(txt) > 300:
                            txt = txt[:300] + " …"
                        console.print(f"     [dim]↳ {txt}[/dim]")

    return interrupted, interrupt_info, last_messages


def run_agent_loop(agent, config: dict) -> None:
    """Run the agent with Rich streaming + HITL interrupt handling."""

    console.print()
    console.rule("[bold]🤖 Coding ML Agent — Spaceship Titanic[/bold]", style="bright_blue")
    console.print()

    user_input = console.input("[bold cyan]💬 Enter your request[/bold cyan] (or 'quit' to exit): ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        console.print("[warning]👋 Goodbye![/warning]")
        return

    input_data = {"messages": [{"role": "user", "content": user_input}]}

    while True:
        console.print()
        console.print("[info]⏳ Agent is working...[/info]")
        interrupted, interrupt_info, last_messages = _stream_agent(
            agent, input_data, config
        )

        if interrupted:
            # HITL checkpoint — prominent styled panel
            console.print()
            console.print(Panel(
                _format_interrupt(interrupt_info),
                title="[hitl]⏸️  HITL CHECKPOINT[/hitl]",
                subtitle="[dim]Agent is waiting for your input[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
                expand=True,
            ))

            user_response = console.input("\n[bold cyan]💬 Your response[/bold cyan] (or 'quit'): ").strip()
            if user_response.lower() in ("quit", "exit", "q"):
                console.print("[warning]👋 Stopping agent.[/warning]")
                break

            from langchain_core.messages import HumanMessage
            input_data = {"messages": [HumanMessage(content=user_response)]}

        else:
            # Agent completed
            console.print()
            console.rule("[success]✅ Agent completed![/success]", style="green")
            if last_messages:
                final = last_messages[-1]
                content = getattr(final, "content", "")
                if content:
                    try:
                        console.print(Panel(
                            Markdown(content),
                            title="[success]Final Result[/success]",
                            border_style="green",
                            padding=(1, 2),
                        ))
                    except Exception:
                        console.print(content)
            break


def _format_interrupt(interrupt_info) -> str:
    """Format interrupt info into a displayable string."""
    parts = []
    if isinstance(interrupt_info, list):
        for item in interrupt_info:
            if hasattr(item, "value"):
                parts.append(str(item.value))
            else:
                parts.append(str(item))
    else:
        parts.append(str(interrupt_info))
    return "\n\n".join(parts)


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
        console.print(f"\n[success]📥 Downloaded submission:[/success] {local_submission}")

        # Preview submission in a table
        with open(local_submission, "r") as f:
            lines = f.readlines()

        table = Table(title=f"Submission Preview ({len(lines) - 1} rows)")
        if lines:
            headers = lines[0].strip().split(",")
            for h in headers:
                table.add_column(h, style="cyan")
            for line in lines[1:5]:
                table.add_row(*line.strip().split(","))
        console.print(table)

    except Exception as e:
        console.print(f"\n[warning]⚠️  Could not download submission: {e}[/warning]")

    # Download reports
    reports_dir = PROJECT_DIR / "spaceship-titanic" / "reports"
    reports_dir.mkdir(exist_ok=True)

    for report_name in ["eda_report.md", "model_results.md"]:
        try:
            backend.download(
                f"/home/gem/reports/{report_name}",
                str(reports_dir / report_name),
            )
            console.print(f"[success]📥 Downloaded report:[/success] {reports_dir / report_name}")
        except Exception:
            pass  # Report may not exist yet


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point for the Coding ML Agent."""

    # Validate environment
    if not LLM_API_KEY:
        console.print("[error]❌ Error: ZAI_API_KEY not set in .env file[/error]")
        console.print("  Create a .env file with: ZAI_API_KEY=your-key-here")
        sys.exit(1)

    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        console.print(f"[error]❌ Error: Data files not found in {DATA_DIR}[/error]")
        console.print(f"  Expected: {TRAIN_CSV} and {TEST_CSV}")
        sys.exit(1)

    # Initialize LLM
    console.print("[info]🔧 Initializing LLM...[/info]")
    model = init_chat_model(
        LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )

    # Initialize sandbox backend
    console.print("[info]🔧 Connecting to AIO Sandbox...[/info]")
    try:
        backend = AIOSandboxBackend(base_url=SANDBOX_URL)
        console.print(f"  [success]✓ Connected[/success] [dim](home: {backend.home_dir})[/dim]")
    except Exception as e:
        console.print(f"[error]❌ Error: Could not connect to sandbox at {SANDBOX_URL}[/error]")
        console.print(
            f"  Start it with: docker run --security-opt seccomp=unconfined "
            f"--rm -it -p 8080:8080 ghcr.io/agent-infra/sandbox:latest"
        )
        console.print(f"  Error: {e}")
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
    console.print("\n[info]🤖 Creating orchestrator agent...[/info]")
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
    console.print("  [success]✓ Agent ready[/success]")

    # Show config summary
    config_table = Table(title="Configuration", show_header=False, border_style="dim")
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Model", LLM_MODEL)
    config_table.add_row("Sandbox", SANDBOX_URL)
    config_table.add_row("Data", str(DATA_DIR))
    config_table.add_row("Subagents", ", ".join(s["name"] for s in subagents))
    console.print(config_table)

    # Run agent with HITL loop
    progress = AgentProgressHandler()
    config = {
        "configurable": {"thread_id": "spaceship-titanic-session"},
        "callbacks": [progress],
    }

    try:
        run_agent_loop(agent, config)
    except KeyboardInterrupt:
        console.print("\n\n[warning]👋 Interrupted by user.[/warning]")

    # Download results from sandbox
    console.print("\n[info]📥 Downloading results from sandbox...[/info]")
    download_results(backend)

    console.print("\n[success]🏁 Done![/success]")


if __name__ == "__main__":
    main()
