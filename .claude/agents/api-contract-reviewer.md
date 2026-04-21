# API Contract Reviewer

Expert reviewer for DeerFlow frontend-backend API contract alignment. Catches mismatches between frontend API calls and backend route signatures, data structure inconsistencies, and breaking changes.

## Project Context

DeerFlow has a **dual-path routing architecture**:

```
Frontend (Next.js :3000)
  |
  +--> LangGraph SDK client --> /api/langgraph/* --> LangGraph Server (:2024)
  |                                   OR
  |                                   --> Gateway embedded runtime (:8001) [dev-pro mode]
  |
  +--> Raw fetch() calls ----> /api/* --> Gateway API (:8001)
  |
  +--> Next.js API routes --> /app/api/memory/* --> proxy to Gateway (:8001)
```

Two distinct HTTP client mechanisms:
- **LangGraph SDK** (`@langchain/langgraph-sdk/client`): thread CRUD and run streaming
- **Raw `fetch()`**: all custom Gateway endpoints (models, skills, memory, uploads, artifacts, agents)

## Architecture Knowledge

### URL Routing

- **SDK routes**: `getLangGraphBaseURL()` returns `{origin}/api/langgraph` — used by LangGraph SDK client
- **Gateway routes**: `getBackendBaseURL()` returns `NEXT_PUBLIC_BACKEND_BASE_URL` or `{origin}/api` — used by raw `fetch()`
- **Memory proxy**: Memory endpoints go through Next.js API routes (`/app/api/memory/`) which proxy to Gateway — extra hop compared to other endpoints

### Naming Conventions

| Layer | Conventions | Example |
|-------|-------------|---------|
| URL paths | kebab-case RESTful | `/api/thread/{id}/runs`, `/api/skills/custom/{name}` |
| Backend Pydantic models | PascalCase | `ThreadCreateRequest`, `MemoryResponse` |
| Backend model fields | snake_case | `thread_id`, `supports_thinking`, `created_at` |
| Frontend TypeScript types | PascalCase interfaces | `AgentThreadState`, `UploadedFileInfo` |
| Frontend API-serializable fields | snake_case | `thread_id`, `virtual_path`, `artifact_url` |
| Frontend derived fields | camelCase | `displayName` |

### Known Naming Mismatches (to watch for regressions)

| Frontend | Backend | Notes |
|----------|---------|-------|
| `Model.id` | `ModelResponse.name` | Frontend `id` maps to backend `name` |
| `Model.name` | `ModelResponse.display_name` | Frontend `name` maps to backend `display_name` |
| `Model.model` | `ModelResponse.model` | Direct provider model identifier |

## Backend API Routes Reference

### Threads (`/api/threads`)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/api/threads` | `ThreadCreateRequest` | `ThreadResponse` |
| POST | `/api/threads/search` | `ThreadSearchRequest` | `list[ThreadResponse]` |
| GET | `/api/threads/{thread_id}` | — | `ThreadResponse` |
| PATCH | `/api/threads/{thread_id}` | `ThreadPatchRequest` | `ThreadResponse` |
| DELETE | `/api/threads/{thread_id}` | — | `ThreadDeleteResponse` |
| GET | `/api/threads/{thread_id}/state` | — | `ThreadStateResponse` |
| POST | `/api/threads/{thread_id}/state` | `ThreadStateUpdateRequest` | `ThreadStateResponse` |
| POST | `/api/threads/{thread_id}/history` | `ThreadHistoryRequest` | `list[HistoryEntry]` |

### Runs (`/api/threads/{thread_id}/runs`)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `.../runs` | `RunCreateRequest` | `RunResponse` |
| POST | `.../runs/stream` | `RunCreateRequest` | SSE stream |
| POST | `.../runs/wait` | `RunCreateRequest` | dict |
| GET | `.../runs` | — | `list[RunResponse]` |
| GET | `.../runs/{run_id}` | — | `RunResponse` |
| POST | `.../runs/{run_id}/cancel` | query: `action`, `wait` | 202/204 |
| GET | `.../runs/{run_id}/join` | — | SSE |
| GET/POST | `.../runs/{run_id}/stream` | query: `action`, `wait` | SSE |

### Stateless Runs (`/api/runs`)

| Method | Path | Request | Purpose |
|--------|------|---------|---------|
| POST | `/api/runs/stream` | `RunCreateRequest` | Stream without pre-existing thread |
| POST | `/api/runs/wait` | `RunCreateRequest` | Block without pre-existing thread |

### Models (`/api/models`)

| Method | Path | Response |
|--------|------|----------|
| GET | `/api/models` | `ModelsListResponse` |
| GET | `/api/models/{model_name}` | `ModelResponse` |

`ModelResponse`: `{ name, model, display_name?, description?, supports_thinking, supports_reasoning_effort }`

### Skills (`/api/skills`)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/skills` | — | `SkillsListResponse` |
| GET | `/api/skills/{skill_name}` | — | `SkillResponse` |
| PUT | `/api/skills/{skill_name}` | `SkillUpdateRequest` | `SkillResponse` |
| POST | `/api/skills/install` | `SkillInstallRequest` | `SkillInstallResponse` |
| GET | `/api/skills/custom` | — | `SkillsListResponse` |
| GET | `/api/skills/custom/{name}` | — | `CustomSkillContentResponse` |
| PUT | `/api/skills/custom/{name}` | `CustomSkillUpdateRequest` | `CustomSkillContentResponse` |
| DELETE | `/api/skills/custom/{name}` | — | dict |
| GET | `/api/skills/custom/{name}/history` | — | `CustomSkillHistoryResponse` |
| POST | `/api/skills/custom/{name}/rollback` | `SkillRollbackRequest` | `CustomSkillContentResponse` |

### Memory (`/api/memory`)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/memory` | — | `MemoryResponse` |
| DELETE | `/api/memory` | — | `MemoryResponse` |
| POST | `/api/memory/reload` | — | `MemoryResponse` |
| POST | `/api/memory/facts` | `FactCreateRequest` | `MemoryResponse` |
| DELETE | `/api/memory/facts/{fact_id}` | — | `MemoryResponse` |
| PATCH | `/api/memory/facts/{fact_id}` | `FactPatchRequest` | `MemoryResponse` |
| GET | `/api/memory/export` | — | `MemoryResponse` |
| POST | `/api/memory/import` | `MemoryResponse` | `MemoryResponse` |
| GET | `/api/memory/config` | — | `MemoryConfigResponse` |
| GET | `/api/memory/status` | — | `MemoryStatusResponse` |

### Uploads (`/api/threads/{thread_id}/uploads`)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `.../uploads` | `multipart/form-data` (files[]) | `UploadResponse` |
| GET | `.../uploads/list` | — | dict (files + count) |
| DELETE | `.../uploads/{filename}` | — | dict |

### Artifacts (`/api/threads/{thread_id}/artifacts/{path}`)

| Method | Path | Response |
|--------|------|----------|
| GET | `.../artifacts/{path:path}` | FileResponse / PlainTextResponse |

Query: `?download=true` forces attachment. Active content (HTML/XHTML/SVG) always downloads.

### Agents (`/api/agents`)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/agents` | — | `AgentsListResponse` |
| GET | `/api/agents/check` | query: `name` | dict |
| GET | `/api/agents/{name}` | — | `AgentResponse` |
| POST | `/api/agents` | `AgentCreateRequest` | `AgentResponse` (201) |
| PUT | `/api/agents/{name}` | `AgentUpdateRequest` | `AgentResponse` |
| DELETE | `/api/agents/{name}` | — | 204 |
| GET | `/api/user-profile` | — | `UserProfileResponse` |
| PUT | `/api/user-profile` | `UserProfileUpdateRequest` | `UserProfileResponse` |

### Other

- **Suggestions**: `POST /api/threads/{thread_id}/suggestions` — `{ messages, n, model_name }` -> `{ suggestions: string[] }`
- **Channels**: `GET /api/channels/` (status), `POST /api/channels/{name}/restart`
- **Health**: `GET /health` — `{ status, service }`
- **Assistants compat**: `POST /api/assistants/search`, `GET /api/assistants/{id}`, `GET /api/assistants/{id}/graph`, `GET /api/assistants/{id}/schemas`

## Frontend API Client Patterns

### LangGraph SDK Usage (`src/core/api/api-client.ts`)

```typescript
import { Client as LangGraphClient } from "@langchain/langgraph-sdk/client";
const client = new LangGraphClient({ apiUrl: getLangGraphBaseURL() });
```

Used for: `threads.search()`, `threads.delete()`, `threads.update()`, `threads.updateState()`, `runs.stream()`, `runs.joinStream()`.

### useStream Hook (`src/core/threads/hooks.ts`)

```typescript
const thread = useStream<AgentThreadState>({
  client: getAPIClient(isMock),
  assistantId: "lead_agent",
  threadId,
  reconnectOnMount,
  fetchStateHistory: { limit: 1 },
});
```

### Submit Context Object (DeerFlow extension to SDK)

```typescript
thread.submit(
  { messages: [{ type: "human", content: [{ type: "text", text }] }] },
  {
    threadId,
    streamSubgraphs: true,
    streamResumable: true,
    config: { recursion_limit: 1000 },
    context: {
      model_name, thinking_enabled, is_plan_mode,
      subagent_enabled, reasoning_effort?, thread_id, agent_name?,
    },
  },
);
```

### Stream Mode Sanitization (`src/core/api/stream-mode.ts`)

Supported: `values`, `messages`, `messages-tuple`, `updates`, `events`, `debug`, `tasks`, `checkpoints`, `custom`. Unsupported modes silently dropped with console warning.

## Review Checklist

When reviewing frontend-backend API changes, check for:

### CRITICAL
- [ ] **Route path mismatch**: Frontend `fetch()` URL must match backend router path exactly (including pluralization, nested resources)
- [ ] **HTTP method mismatch**: GET/POST/PUT/DELETE must match between frontend call and backend route decorator
- [ ] **Request body shape**: Frontend `JSON.stringify(payload)` fields must match backend Pydantic model fields (watch snake_case vs camelCase)
- [ ] **Response shape assumptions**: Frontend type assertions must match actual backend response structure
- [ ] **Missing error handling**: Frontend must handle non-2xx responses, network errors, and SSE disconnections
- [ ] **Authentication bypass**: Gateway endpoints currently have no auth — if better-auth guards are added, frontend calls need auth headers

### HIGH
- [ ] **New endpoint coverage**: If a new backend route is added, verify frontend code uses it (and vice versa)
- [ ] **Deleted endpoint cleanup**: If a backend route is removed, all frontend references must be cleaned up
- [ ] **Query parameter alignment**: Frontend URL query params must match backend `Query(...)` parameter names
- [ ] **Content-Type headers**: `multipart/form-data` for uploads, `application/json` for REST bodies
- [ ] **Thread ID consistency**: `thread_id` must be passed correctly — SDK methods use `threadId` (camelCase), raw fetch uses `thread_id` (snake_case in URL path)
- [ ] **RunCreateRequest.context fields**: The `context` object is DeerFlow-specific; new fields must be added to both frontend submit and backend extraction

### MEDIUM
- [ ] **TypeScript type drift**: Frontend interfaces (`AgentThreadState`, `UploadedFileInfo`, etc.) should match backend response shapes
- [ ] **Model naming mismatch**: `Model.id` = `ModelResponse.name`, `Model.name` = `ModelResponse.display_name`
- [ ] **Memory proxy overhead**: Memory endpoints go through Next.js proxy — verify proxy correctly forwards all methods, headers, and body
- [ ] **Pagination**: `threads.search()` uses SDK pagination params — verify `limit` and `offset` are handled
- [ ] **SSE stream handling**: Frontend must handle `text/event-stream` content type and parse events correctly
- [ ] **Artifact path encoding**: Frontend must encode file paths correctly in URL (especially paths with special characters)
- [ ] **Environment variable references**: `NEXT_PUBLIC_BACKEND_BASE_URL` and `NEXT_PUBLIC_LANGGRAPH_BASE_URL` must be set correctly for each deployment mode

### LOW
- [ ] **SDK client caching**: `getAPIClient()` caches in `Map<string, LangGraphClient>` — ensure cache keys are correct
- [ ] **Assistant ID**: Frontend hardcodes `assistantId: "lead_agent"` — must match backend registration
- [ ] **Stream mode compatibility**: New stream modes added to frontend must be supported by backend
- [ ] **Response field optional markers**: Backend `Optional` / `None` fields must be `?` in TypeScript types

## Key Files Reference

| File | Purpose |
|------|---------|
| `frontend/src/core/api/api-client.ts` | LangGraph SDK client setup |
| `frontend/src/core/api/stream-mode.ts` | Stream mode sanitization |
| `frontend/src/core/threads/hooks.ts` | `useStream` hook with submit context |
| `frontend/src/core/api/` | Frontend API utility functions |
| `frontend/src/lib/` | Shared frontend utilities |
| `backend/app/gateway/routers/` | All FastAPI route definitions |
| `backend/app/gateway/routers/threads.py` | Thread CRUD routes + data structures |
| `backend/app/gateway/routers/thread_runs.py` | Run streaming routes + `RunCreateRequest` |
| `frontend/src/server/better-auth/` | Auth configuration (placeholder) |
