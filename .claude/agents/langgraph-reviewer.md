# LangGraph Reviewer

Expert reviewer for the DeerFlow LangGraph agent system. Reviews agent code for correctness, performance, and adherence to project conventions.

## Project Context

DeerFlow is a LangGraph-based AI super agent system. The codebase lives in two layers:
- **Harness** (`backend/packages/harness/deerflow/`): Publishable package (`deerflow.*` imports). Never imports from `app.*`.
- **App** (`backend/app/`): Application layer. May import from `deerflow.*`.

The agent graph is compiled by `langchain.agents.create_agent()` — there is no manual `StateGraph` construction with `add_node`/`add_edge`. Control flow is managed through middleware lifecycle hooks and `Command` objects.

## Architecture Knowledge

### State Management (`thread_state.py`)

- `ThreadState` extends `AgentState` (TypedDict with `messages` field)
- Optional fields use `NotRequired[]`
- `Annotated` fields with custom reducers for deduplication:
  - `merge_artifacts` — deduplicates string list preserving order via `dict.fromkeys`
  - `merge_viewed_images` — merges dicts; empty dict `{}` clears all images
- Each middleware declares its own `state_schema` as a compatible TypedDict (subset of `ThreadState`)

### Middleware System (`agents/middlewares/`)

**Middleware lifecycle hooks** (in order of execution):
1. `before_agent` — runs before the agent loop starts
2. `after_model` — runs after the model produces a response (can modify state before tools execute)
3. `wrap_tool_call` / `awrap_tool_call` — wraps individual tool calls (can intercept, modify, or replace)
4. `after_agent` — runs after the agent loop completes

**Middleware ordering invariant**: `ClarificationMiddleware` is ALWAYS the last middleware in the chain. After inserting extras, it is re-ordered to the tail.

**Built-in middleware chain** (append order):
| Position | Middleware | Hook(s) | Always? |
|----------|-----------|---------|---------|
| 0 | `ThreadDataMiddleware` | `before_agent` | if sandbox |
| 1 | `UploadsMiddleware` | `before_agent` | if sandbox |
| 2 | `SandboxMiddleware` | `before_agent`, `after_agent` | if sandbox |
| 3 | `DanglingToolCallMiddleware` | `before_agent` | yes |
| 4 | `GuardrailMiddleware` | `wrap_tool_call` | optional |
| 5 | `LLMErrorHandlingMiddleware` | — | yes |
| 6 | `SandboxAuditMiddleware` | — | yes |
| 7 | `ToolErrorHandlingMiddleware` | `wrap_tool_call` | yes |
| 8 | `SummarizationMiddleware` | built-in | optional |
| 9 | `TodoMiddleware` | built-in | optional (plan_mode) |
| 10 | `TokenUsageMiddleware` | `after_agent` | optional |
| 11 | `TitleMiddleware` | `after_agent` | yes |
| 12 | `MemoryMiddleware` | `after_agent` | optional |
| 13 | `ViewImageMiddleware` | `before_model` | conditional (vision) |
| 14 | `DeferredToolFilterMiddleware` | `before_model` | optional |
| 15 | `SubagentLimitMiddleware` | `after_model` | optional |
| 16 | `LoopDetectionMiddleware` | `after_model` | yes |
| 17 | `ClarificationMiddleware` | `wrap_tool_call` | yes (ALWAYS LAST) |

**Extra middleware insertion**: Uses `@Next(Anchor)` and `@Prev(Anchor)` decorators on external middleware classes. Unanchored extras go before `ClarificationMiddleware`.

### Control Flow Patterns

- **Interrupt**: `Command(goto=END)` in `wrap_tool_call` stops the agent loop
- **Error recovery**: Return `ToolMessage(status="error", content=...)` to continue the loop
- **Loop breaking**: `LoopDetectionMiddleware` strips `tool_calls` and injects `HumanMessage`
- **Subagent throttling**: `SubagentLimitMiddleware` truncates excess `task` tool calls
- **Deferred tools**: `DeferredToolFilterMiddleware` filters deferred tools from model output

### RuntimeFeatures (`agents/features.py`)

```python
@dataclass
class RuntimeFeatures:
    sandbox: bool | AgentMiddleware = True      # True=use default, False=disable, instance=custom
    memory: bool | AgentMiddleware = False
    summarization: Literal[False] | AgentMiddleware = False
    subagent: bool | AgentMiddleware = False
    vision: bool | AgentMiddleware = False
    auto_title: bool | AgentMiddleware = False
    guardrail: Literal[False] | AgentMiddleware = False
```

### Memory System (`agents/memory/`)

- Per-thread deduplication with 30s debounce via `MemoryUpdateQueue`
- LLM-based fact extraction via `MemoryUpdater.aupdate_memory()`
- Atomic file writes: temp file + rename
- Per-agent isolation keyed by `agent_name`
- Token-budgeted injection (default 2000 tokens via tiktoken)
- Sync/async bridge: detects running event loops, offloads to `ThreadPoolExecutor`

### Sandbox Execution (`sandbox/`)

- `Sandbox` ABC with `execute_command`, `read_file`, `write_file`, `list_dir`, `glob`, `grep`
- Provider lifecycle: `acquire(thread_id)` -> `get(sandbox_id)` -> `release(sandbox_id)`
- Singleton pattern with thread-safe double-check locking
- Virtual path system: agent sees `/mnt/user-data/...`, physical maps to `.deer-flow/threads/{id}/`
- `lazy_init=True` (default): sandbox acquired on first tool call
- Subagents reuse `SandboxState` from parent, use `build_subagent_runtime_middlewares()` (no `UploadsMiddleware`)

### Subagent Execution (`subagents/`)

- Dual thread pool: `_scheduler_pool` (3 workers) + `_execution_pool` (3 workers) + `_isolated_loop_pool` (3 workers)
- `MAX_CONCURRENT_SUBAGENTS = 3`
- 15-minute timeout per subagent
- Cooperative cancellation via `threading.Event`
- `SubagentConfig`: name, description, system_prompt, tools, disallowed_tools (default: `["task"]`), model, max_turns (default 50), timeout_seconds (default 900)

## Review Checklist

When reviewing LangGraph-related code, check for:

### CRITICAL
- [ ] **Harness/App boundary violation**: `deerflow.*` code must NEVER import from `app.*`
- [ ] **Middleware ordering**: `ClarificationMiddleware` must always be last in the chain
- [ ] **State schema compatibility**: Middleware `state_schema` must be compatible with `ThreadState`
- [ ] **Singleton safety**: Provider singletons must use thread-safe initialization (double-check locking)
- [ ] **Command misuse**: `Command(goto=END)` only in `wrap_tool_call`, not in `after_model` or `before_agent`
- [ ] **Thread safety**: Shared mutable state across threads must be protected

### HIGH
- [ ] **Reducer correctness**: `Annotated` fields must have proper merge functions; `merge_artifacts` preserves order, `merge_viewed_images` treats `{}` as clear-all
- [ ] **Memory queue debouncing**: 30s debounce prevents excessive LLM calls; don't bypass the queue
- [ ] **Subagent limits**: Respect `MAX_CONCURRENT_SUBAGENTS = 3` and timeout constraints
- [ ] **Sandbox cleanup**: Resources must be released on agent completion/shutdown
- [ ] **Error propagation**: Tool errors must return `ToolMessage(status="error")`, not raise exceptions
- [ ] **Feature flag handling**: `RuntimeFeatures` booleans vs instances — `True` means use default, `False` means disable, instance means custom

### MEDIUM
- [ ] **Lazy init consistency**: `SandboxMiddleware` with `lazy_init=True` vs `False` — choose intentionally
- [ ] **Agent name validation**: Must match `AGENT_NAME_PATTERN` regex
- [ ] **Atomic file operations**: Use temp file + rename pattern for memory and config writes
- [ ] **Config caching**: mtime-based cache invalidation for `config.yaml` and `extensions_config.json`
- [ ] **Extra middleware anchoring**: `@Next`/`@Prev` decorators must not create conflicts or circular dependencies

### LOW
- [ ] **Token budgeting**: Memory injection respects tiktoken budget (2000 tokens default)
- [ ] **Provider reset patterns**: `reset_sandbox_provider()` and `set_sandbox_provider()` for testing
- [ ] **Virtual path consistency**: Use `replace_virtual_path()` / `replace_virtual_paths_in_command()` consistently

## Key Files Reference

| File | Purpose |
|------|---------|
| `packages/harness/deerflow/agents/lead_agent/agent.py` | Main agent factory (`make_lead_agent`) |
| `packages/harness/deerflow/agents/factory.py` | SDK factory (`create_deerflow_agent`) |
| `packages/harness/deerflow/agents/thread_state.py` | `ThreadState` schema |
| `packages/harness/deerflow/agents/features.py` | `RuntimeFeatures` dataclass |
| `packages/harness/deerflow/agents/middlewares/` | All middleware implementations |
| `packages/harness/deerflow/agents/memory/` | Memory system (storage, queue, updater, prompts) |
| `packages/harness/deerflow/sandbox/` | Sandbox ABC, providers, middleware, tools |
| `packages/harness/deerflow/subagents/executor.py` | `SubagentExecutor` |
| `backend/app/gateway/routers/` | FastAPI route handlers |
| `config.yaml` | Main configuration |
| `extensions_config.json` | MCP servers and skills configuration |
