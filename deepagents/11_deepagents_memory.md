# Quản lý Bộ nhớ (Memory) trong DeepAgents

Việc thêm bộ nhớ vĩnh viễn (persistent memory) vào các agent được xây dựng bằng DeepAgents giúp chúng có khả năng học hỏi và tự cải thiện qua từng cuộc trò chuyện.

## 1. Cách bộ nhớ hoạt động (How memory works)

1. **Trỏ agent tới các file bộ nhớ (Point the agent at memory files):**
   Truyền đường dẫn file vào tham số `memory=` khi tạo agent. Bạn cũng có thể truyền [skills](./skills.md) (các kỹ năng) thông qua tham số `skills=` cho "procedural memory" (những tập lệnh có thể tái sử dụng hướng dẫn agent cách thực hiện tác vụ). Một [backend](./07_deepagents_backends.md) sẽ kiểm soát nơi lưu trữ file và ai có thể truy cập chúng.
   
2. **Agent đọc bộ nhớ (Agent reads memory):**
   Agent có thể tải các file bộ nhớ vào *system prompt* ngay lúc khởi động, hoặc đọc chúng khi cần thiết (on demand) trong suốt cuộc trò chuyện. Ví dụ: Các `skills` sử dụng cơ chế on-demand, agent chỉ đọc mô tả ngắn của kỹ năng lúc khởi động và chỉ tải nội dung chi tiết nếu kỹ năng được dùng định khớp với tác vụ. Điều này giúp ngữ cảnh (context) luôn ngắn gọn.
   
3. **Agent cập nhật bộ nhớ - Tùy chọn (Agent updates memory):**
   Khi nhận biết thông tin mới, agent có thể dùng công cụ tích hợp sẵn `edit_file` để cập nhật lại các file bộ nhớ. Việc cập nhật có thể thực hiện tức thời trong lúc chat (mặc định) hoặc chạy ngầm giữa các cuộc hội thoại (Background consolidation). Những thay đổi này sẽ được lưu giữ lại ở những phiên làm việc tiếp theo.
   *Lưu ý:* Không phải bộ nhớ nào cũng cho phép ghi: Các skills định nghĩa sẵn và các chính sách của tổ chức (organization policies) thường ở dạng chỉ đọc (read-only).

---

## 2. Các phạm vi của bộ nhớ (Scoped memory)

### 2.1 Bộ nhớ theo phạm vi Agent (Agent-scoped memory)
Đôi khi, bạn muốn phân lập bộ nhớ cho từng loại agent cụ thể (ví dụ: một agent làm code web, một agent viết blog sẽ có file bộ nhớ riêng). 

```python
from langchain_core.utils.uuid import uuid7
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Seed file bộ nhớ ban đầu
store.put(
    ("my-agent",),
    "/memories/AGENTS.md",
    create_file_data("""## Response style
- Keep responses concise
- Use code examples where possible
"""),
)

# Seed một kỹ năng (skill)
store.put(
    ("my-agent",),
    "/skills/langgraph-docs/SKILL.md",
    create_file_data("""---
name: langgraph-docs
description: Fetch relevant LangGraph documentation to provide accurate guidance.
---
# langgraph-docs
Use the fetch_url tool to read https://docs.langchain.com/llms.txt, then fetch relevant pages.
"""),
)

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro",
    memory=["/memories/AGENTS.md"],
    skills=["/skills/"],
    backend=lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={
            "/memories/": StoreBackend(
                rt, namespace=lambda rt: ("my-agent",)
            ),
            "/skills/": StoreBackend(
                rt, namespace=lambda rt: ("my-agent",)
            ),
        },
    ),
    store=store,
)
```

### 2.2 Bộ nhớ theo phạm vi User (User-scoped memory)
Bạn có thể cô lập bộ nhớ dựa trên `user_id`, giúp agent có thể có trí nhớ riêng biệt với từng người dùng độc lập. Điều này cực kỳ hữu ích cho ứng dụng public:

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro",
    memory=["/memories/preferences.md"],
    skills=["/skills/"],
    backend=lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={
            "/memories/": StoreBackend(
                rt, namespace=lambda rt: (rt.server_info.user.identity,)
            ),
            "/skills/": StoreBackend(
                rt, namespace=lambda rt: (rt.server_info.user.identity,)
            ),
        },
    ),
    store=store,
)
```

---

## 3. Cách sử dụng Nâng cao (Advanced usage)

### 3.1 Bộ nhớ theo từng khóa trò chuyện (Episodic memory)
Agent thường chỉ chú ý đến thông tin trong một luồng chat hiện tại. Tuy nhiên, bạn có thể tạo bộ nhớ theo tập (Episodic) bằng cách giúp agent tra cứu lại các sự kiện ở quá khứ:

```python
from langgraph_sdk import get_client
from langchain.tools import tool, ToolRuntime

client = get_client(url="<DEPLOYMENT_URL>")

@tool
async def search_past_conversations(query: str, runtime: ToolRuntime) -> str:
    """Search past conversations for relevant context."""
    user_id = runtime.server_info.user.identity
    threads = await client.threads.search(
        metadata={"user_id": user_id},
        limit=5,
    )
    
    results = []
    for thread in threads:
        history = await client.threads.get_history(thread_id=thread["thread_id"])
        results.append(history)
        
    return str(results)
```

### 3.2 Bộ nhớ cấp độ tổ chức (Organization-level memory)
Nếu agent làm việc trong một SaaS (B2B), bạn có thể gom nhóm bộ nhớ dành cho từng công ty/tổ chức (Organization) đó và kết hợp với chính sách chung (ví dụ quy định, chính sách bảo hiểm, v.v.).

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro",
    memory=[
        "/memories/preferences.md", 
        "/policies/compliance.md",
    ],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
            "/policies/": StoreBackend(
                namespace=lambda rt: (rt.context.org_id,),
            ),
        },
    ),
)
```

### 3.3 Tổng hợp bộ nhớ chạy ngầm (Background consolidation)
Để bộ nhớ file không bị phình to làm lãng phí token mỗi khi chat, bạn có thể tạo một "Consolidate Agent" riêng chạy ngầm (ví dụ: thông qua một Cron job trên cấu hình `langgraph.json`) có nhiệm vụ tóm tắt thông tin người dùng từ các hội thoại gần đây và ghi đè nội dung file bộ nhớ cho ngắn gọn hơn.

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro",
    system_prompt="""Review recent conversations and update the user's memory file. 
Merge new facts, remove outdated information, and keep it concise.""",
    tools=[search_recent_conversations],
)
```

**Cron schedule** trên LangSmith:
```python
cron_job = await client.crons.create(
    assistant_id="consolidation_agent",
    schedule="0 */6 * * *",
    input={"messages": [{"role": "user", "content": "Consolidate recent memories."}]},
)
```

---

## 4. Thực tiễn áp dụng và cấu hình chuẩn cho `deepagents==0.5.3` (Best Practices)

Từ dữ liệu tìm kiếm hệ thống liên quan đến thư viện `deepagents==0.5.3`, khi thiết kế ứng dụng có quản lý bộ nhớ, bạn cần chú ý các điểm cốt lõi sau:

### 1. Phân biệt cách Middleware tải "Memory" và "Skills"
- `memory=`: Các tập tin cung cấp ở đây (ví dụ: `["./AGENTS.md"]`) sẽ được Middleware **tải trực tiếp vào System Prompt** của Agent. Điều này rất thích hợp với quy định chung và bối cảnh (context) nhỏ gọn.
- `skills=`: Thư mục truyền vào đây (ví dụ: `["./skills/"]`) sẽ được Middleware **tải theo yêu cầu (on demand)**. Điều này giúp tối ưu số lượng token context xử lý cho LLM.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    memory=["./AGENTS.md"],           # ← Middleware loads directly into system prompt
    skills=["./skills/"],             # ← Middleware loads on demand
    backend=FilesystemBackend(root_dir="./") 
)
```

### 2. Các điểm cần chú ý về "Bộ nhớ chỉ đọc" (Read-only vs Writable memory gotchas)
Mặc định trong môi trường DeepAgents:
- Tệp cấu hình gốc `AGENTS.md` (hoặc bất kỳ tệp memory nào nằm ở root/ứng dụng) và toàn bộ cấu trúc thư mục `skills/` **ĐỀU LÀ CHỈ ĐỌC (Read-only)**. Agent không thể vô ý ghi đè hệ thống trong lúc runtime (trừ khi có sự can thiệp từ developer cập nhật lại file và deploy).
- Để Agent có khả năng "nhớ" và lưu trữ (dùng tool `edit_file`), bạn phải cung cấp vị trí thư mục rõ ràng và có thể ghi được (ví dụ `/memories/user/AGENTS.md`) thông qua cấu hình `StoreBackend`.

### 3. Giải quyết Ghi đồng thời (Concurrent Writes)
- Việc cho phép agent cập nhật memory trực tiếp trên cuộc trò chuyện có thể gây ra conflict (xung đột dữ liệu file) nếu người dùng có hai luồng chat đồng thời thao tác ghi file thay đổi bộ nhớ.
> **Khuyến nghị**: Đối với Production, tốt nhất là cô lập bộ nhớ thay đổi thông qua một *Consolidation agent* chạy nền để đọc qua các đoạn chat gần nhất và làm mới lại file bộ nhớ thay vì để agent hiện tại tự ghi đè từng chút một.
