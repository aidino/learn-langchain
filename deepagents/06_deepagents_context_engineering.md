# Context Engineering trong Deep Agents (Quản lý Context - Ngữ cảnh)

Tài liệu này giải thích cách kiểm soát những dữ liệu (context) mà tác tử (Deep Agent) có thể truy cập và cách framework quản lý phần context đó xuyên suốt các chuỗi tác vụ dài hạn.

Dưới đây là bản dịch và diễn giải chi tiết từ tài liệu [Context engineering in Deep Agents](https://docs.langchain.com/oss/python/deepagents/context-engineering).

---

## 1. Các loại Context (Types of Context)

Hệ thống Deep Agents phân loại context ra thành nhiều nhóm để dễ quản lý:
- **Input context (Context đầu vào)**: Các thông tin được cung cấp ngay từ khi khởi tạo agent (ví dụ: system prompt, memory, skills).
- **Runtime context (Context thời gian thực)**: Dữ liệu được truyền vào tại thời điểm hàm chạy (invoke/ainvoke) như `user_id` hay `api_key`.
- **Context compression (Nén context)**: Các kỹ thuật giảm tải độ dài của lịch sử trò chuyện để tránh vượt mức giới hạn token của LLM.
- **Context isolation (Cách ly context)**: Việc sử dụng các Tác tử con (Subagent) để bảo vệ context của tác tử chính khỏi bị nhiễu do các thông tin quá rườm rà.
- **Long-term memory (Bộ nhớ dài hạn)**: Khả năng lưu trữ và truy xuất thông tin người dùng / kiến thức qua nhiều phiên trò chuyện khác nhau.

---

## 2. Input Context (Context Cố định Đầu vào)

Đây là những thông tin cốt lõi định hình hành vi và cung cấp kiến thức nền cho agent. Chúng sẽ được nạp ngay khi khởi tạo agent.

### 2.1. System Prompt (Lời nhắc hệ thống chủ)

Bạn có thể truyền trực tiếp chuỗi lệnh thông qua `system_prompt` khi tạo agent:
```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    system_prompt=(
        "You are a research assistant specializing in scientific literature. "
        "Always cite sources. Use subagents for parallel research on different topics."
    ),
)
```
**Giải thích:** `system_prompt` này sẽ được gộp chung với các cấu trúc prompt nội bộ ẩn của DeepAgent để hình thành nên "Complete System Prompt" cuối cùng.

### 2.2. Memory (File cấu hình nguyên tắc nhóm)

Thay vì nhồi nhét tất cả nguyên tắc vận hành vào biến chuỗi `system_prompt`, bạn có thể chỉ định các file tài liệu bằng cấp thông qua tham số `memory`. Giúp dễ dàng đồng bộ nguyên tắc nhóm (`AGENTS.md`) hoặc tuỳ chọn người dùng.
```python
agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    memory=["/project/AGENTS.md", "~/.deepagents/preferences.md"],
)
```

### 2.3. Skills (Kỹ năng chia nhỏ)

Thay vì trang bị hàng chục công cụ (tool) phức tạp ở mọi lúc, Agent có thể được nạp thêm các thư mục mã nguồn chứa `skills`. Prompt hệ thống sẽ tự tìm cách hướng dẫn Agent gọi ra các skill này khi cần:
```python
agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    skills=["/skills/research/", "/skills/web-search/"],
)
```

### 2.4. Tool Prompts (Các nhắc nhở về công cụ mặc định)

Deep Agents sẽ tự động bổ sung prompt vào hệ thống để hướng dẫn mô hình cách dùng những công cụ tích hợp sẵn:
- **Planning prompt:** Hướng dẫn cách lập kế hoạch công việc (dùng tool `write_todos`).
- **Filesystem prompt:** Cách dùng filesystem ảo như `ls`, `read_file`, `write_file`, v.v. (và `execute` nếu sandbox được bật).
- **Subagent prompt:** Cách giao việc thông qua công cụ `task`.
- **Human-in-the-loop prompt:** Khi nào cần dừng lại để hỏi ý kiến người dùng nếu cấu hình `interrupt_on` được kích hoạt.

### 2.5. Cấu tạo Complete System Prompt (Toàn bộ Prompt hệ thống)

Một LLM cuối cùng sẽ nhận được một hướng dẫn (Prompt) cấu thành từ 9 mảnh ghép theo thứ tự:
1. `system_prompt` (do bạn tự viết).
2. Lời nhắc cơ bản của Agent (Base agent prompt).
3. Lời nhắc lên kế hoạch To-do list.
4. Lời nhắc Memory (từ các file truyền qua `memory`).
5. Lời nhắc về Skills (tóm tắt các skill và nơi chứa nó).
6. Lời nhắc Virtual filesystem (Hệ thống file ảo).
7. Lời nhắc Subagent (Cách uỷ quyền).
8. Các thông tin bổ sung từ custom Middleware.
9. Lời nhắc tương tác Human-in-the-loop.

---

## 3. Runtime Context (Context Thời gian thực)

Khi tạo các công cụ (Tool) tuỳ chỉnh, rất có thể bạn sẽ cần những thông tin như `api_key` hoặc thông tin của một người dùng đăng nhập hiện hành (`user_id`). Thay vì nhồi nhúng những chuỗi bảo mật này vào System Prompt (nơi có nguy cơ rò rỉ mã), bạn nên truyền chúng động tại runtime.

**Giải pháp:** Định nghĩa `context_schema`, khai báo tham số tự động nạp `runtime: ToolRuntime[Context]` bên trong tool, và truyền data mảng lúc gọi `invoke`.

```python
from dataclasses import dataclass
from deepagents import create_deep_agent
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str
    api_key: str

@tool
def fetch_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """Lấy dữ liệu của user hiện tại."""
    # Lấy ID của user một cách động, an toàn mà không cần truyền vào prompt
    user_id = runtime.context.user_id
    text = f"Dữ liệu trả về của mã {user_id}: {query}"
    return text

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    tools=[fetch_user_data],
    context_schema=Context,
)

# Chạy thực tế tác vụ, gắn với user Context hiện tại
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Tìm kiếm hoạt động của tôi"}]},
    context=Context(user_id="user-123", api_key="sk-xxxx..."),
)
```

---

## 4. Context Compression (Nén Context ngăn tràn bộ nhớ)

Việc giữ mọi tương tác, log code, error trong luồng hội thoại chung sẽ khiến agent nhanh chóng bị cạn kiệt số lượng token hạn mức. Ở hệ sinh thái DeepAgents, vấn đề này được giải quyết êm đẹp bằng 2 cơ chế:

### 4.1. Offloading (Đẩy dữ liệu "mập Mạp" ra ngoài Filesystem)
- Nếu một **Đầu vào hay Đầu ra của Tool mà dài quá giới hạn 20.000 token**, hệ thống sẽ KHÔNG chèn thẳng văn bản đó vào danh sách `messages`.
- Thay vào đó, DeepAgents tự động *ghi nội dung khổng lồ đó thành một file trên ổ đĩa ảo*. Tại thanh lịch sử chat, agent chỉ nhận lại một văn bản tham chiếu gồm **Đường dẫn file (FilePath) + 10 dòng đầu tiên (Preview)**. Agent nào gặp cần đọc chi tiết đều sẽ phải gọi tool `read_file` sau. Chết chế này giảm thiểu tối đa sự phình to context.

### 4.2. Summarization (Tóm tắt chuỗi hội thoại cũ vĩnh viễn)
- Bất cứ khi nào Agent đã tiệu thụ token lên tới mức **85%** sức chứa mặc định của `max_input_tokens`, hoặc một API call tới LLM lỡ gặp lỗi đặc biệt `ContextOverflowError`:
  1. Tiến trình chat sẽ tạm dừng.
  2. Một LLM phụ sẽ nhận toàn bộ hội thoại và tiến hành tạo bản tóm tắt chiến lược.
  3. Bản trò chuyện đầy đủ sẽ được lưu về ổ đĩa ảo để giữ bản sao (Filesystem preservation).
  4. Context mà Agent đang làm việc chỉ còn giữ lại **1 Đoạn Tóm Tắt + 10% các tin nhắn hoạt động mới nhất**.

*Để kích hoạt Tool này, có thể gọi middleware riêng biệt:*
```python
from deepagents.middleware.summarization import create_summarization_tool_middleware

agent = create_deep_agent(
    model=model,
    middleware=[
        create_summarization_tool_middleware(model, backend),
    ],
)
```

---

## 5. Context Isolation (Cách ly dội âm với Subagents)

Đừng để agent chính mất tập trung khi nó phải đi tìm và duyệt qua hàng tá tài liệu web nhảm nhí.
- **Sử dụng Tool `task`**: Tác tử tổng quản có thể dùng lệnh `task` để tạo ra một Tác tử con (Subagent).
- **Môi trường kín**: Phân luồng công việc cho phép tác tử con tư duy cực kỳ sâu sắc trong một mưu đồ hội thoại khác (Fresh context) mà tiến trình mẹ không hay biết gì.
- **Báo cáo sạch gọn**: Hệ thống yêu cầu tác tử con TỪ CHỐI nhồi luồng tìm kiếm lại, mà chỉ được đưa lại một *Báo cáo cực kỳ cốt thép (final report)* cho ông tiến trình mẹ, giữ cho Context của mẹ hoàn toàn sạch.

---

## 6. Long-term Memory (Bộ nhớ dài hạn đa phiên bản)

Làm sao để agent nhớ "Người dùng nay không thích ăn cà chua" đi từ Thread này qua hẳn Thread sau? Cần một cơ sở hạ tầng backend phức hợp (`CompositeBackend`).

**Giải pháp:** Mọi thứ được lưu ảo (chỉ tồn tại ở thread) qua `StateBackend`. Trừ một đường dẫn ảo cố định tên là `/memories/`. Nếu agent trỏ vào ghi thư mục này, nội dung sẽ được định tuyến xuống một backend vĩnh viễn là `StoreBackend` (ví dụ database Postgres, BaseMemory).

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

def make_backend(runtime):
    return CompositeBackend(
        # Luôn làm mới ảo
        default=StateBackend(runtime),
        # Nếu đường dẫn đụng tới memories, đưa nó ghi thẳng vĩnh viễn
        routes={"/memories/": StoreBackend(runtime)},
    )

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    store=InMemoryStore(),
    backend=make_backend,
    system_prompt='Khi người dùng chia sẻ sở thích cá nhân của họ, bạn phải gọi Tool write_file lưu vào "/memories/user_preferences.txt".',
)
```
Bằng file này, ở những cuộc nói chuyện của 10 ngày sau, tác tử vẫn cứ mò về `read_file` thư mục cấu hình cá nhân này và sẽ làm vừa ý bạn.

---

## 7. Best Practices Hội Tụ

1. **Bắt đầu bằng một Input tối giản và hiệu quả**: Dành vị trí `memory` cho các nguyên tắc vận hành team dài hạn; dùng `skills` khi có tình huống cá biệt, không nhồi chung hết vào 1 cục.
2. **Uỷ quyền cho Tác tử con cường độ cao**: Nhượng quyền các quy trình đa bước phức tạp cho mạng lưới Tác tử con.
3. **Bóp nghẹt độ dài ra của Subagent**: Nếu thấy tác tử con đưa ra kết quả rườm rà, hãy sửa system_prompt của nó thêm câu ép buộc báo cáo dưới 500 chữ.
4. **Việc có Filesystem, Tái sử dụng**: Hãy offload các output khổng lồ bằng file. Dạy tác tử đọc từng cụm dữ liệu thay vì load cục bộ lên memory.
5. **Hướng dẫn rành mạch file dài hạn `/memories/`**: Quy định thẳng chuẩn cấu trúc và cách ghi thư mục bộ nhớ dài hạn trong system component để duy trì sự phát triển lâu bền cho agent.
6. **Pass biến Session qua Tool Runtime**: Luôn truyền Context State Schema API key, Cấu hình session để tránh dính security bug trên Log.

---

## 8. Ví dụ Thực Tế: Research Agent (DeepAgents v0.5.3)

Dưới đây là một ví dụ kết hợp chuẩn xác các thực hành tốt nhất về thiết kế ngữ cảnh để tạo ra một **Research Agent** đa nhiệm theo chuẩn bản cập nhật `0.5.3`. 
Ví dụ này trình diễn việc sử dụng `subagents` với ngữ cảnh cách ly (Context Isolation) và công cụ nạp động nhằm giữ sạch memory của tác tử Tổng quản:

```python
from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def tavily_search(query: str) -> str:
    """Tìm kiếm web sử dụng Tavily API để lấy thông tin mới nhất."""
    return f"Kết quả tìm kiếm cho: {query}"

@tool
def think(thought: str) -> str:
    """Ghi lại suy nghĩ chiến lược hoặc phản ánh về quy trình nghiên cứu."""
    return f"Đã ghi lại: {thought}"

current_date = datetime.now().strftime("%Y-%m-%d")

# 1. Tác tử con (Subagent) nghiên cứu - Ngữ cảnh hoàn toàn độc lập
research_sub_agent = {
    "name": "research-agent",
    "description": "Giao phó việc nghiên cứu cho tác tử này. Cho nó 1 chủ đề mỗi lần để điều tra sâu.",
    "system_prompt": f"""Bạn là một chuyên gia nghiên cứu. Hôm nay là ngày {current_date}.

Nhiệm vụ của bạn là nghiên cứu kỹ lưỡng chủ đề được giao bằng công cụ tìm kiếm web.
- Dùng tavily_search để thu thập chi tiết.
- Dùng think để ghi lại tiến trình.
- TỔNG HỢP phát hiện thành một báo cáo rành mạch, KHÔNG trả về raw data.
- Bắt buộc phải gắn kèm trích dẫn (citations).""",
    "tools": [tavily_search, think],
}

# 2. Ngữ cảnh của Tác tử Điều phối (Orchestrator) 
ORCHESTRATOR_PROMPT = """Bạn là một Tổng quản lý dự án (Orchestrator) cho các dự án nghiên cứu phức tạp.

Khi nhận được yêu cầu nghiên cứu:
1. Chia nhỏ nó thành các câu hỏi cụ thể.
2. Giao phó từng câu hỏi cho tác tử con `research-agent`.
3. Có thể khởi chạy chạy song song nhiều agents nếu các câu hỏi độc lập với nhau.
4. Tổng hợp TẤT CẢ báo cáo con thành một Báo cáo Cuối Cùng chuẩn xác.

Sử dụng tool `task` để giao việc. Tối đa 3 tác vụ nghiên cứu đồng thời."""

model = init_chat_model("anthropic:claude-sonnet-4-6", temperature=0.0)

# Khởi tạo Main Agent kế thừa các nguyên tắc
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think],
    system_prompt=ORCHESTRATOR_PROMPT,
    subagents=[research_sub_agent],
)

# Kích hoạt Agent - Context Memory đang được bảo vệ khỏi 'rác' web data
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Nghiên cứu về tình trạng hiện tại của LLMs, tập trung vào khả năng reasoning và multimodal."}
    ]
})

print(result["messages"][-1].content)
```

**Biểu hiện của Context Engineering trong ví dụ trên:**
- **Dynamic Context Injection:** Tiêm trực tiếp `current_date` vào prompt để LLM biết thời gian hệ thống mà không cần fix cứng, giữ cho state linh hoạt.
- **Context Isolation:** Subagent nhận lệnh qua tool `task`, làm những việc "dơ bẩn" và tốn token nhất như xài tool search web -> sau đó trả về một báo cáo đã tóm tắt, tự động cắt bỏ context thừa thải.
- **Ngăn Context Pollution (Ô nhiễm ngữ cảnh):** Tiến trình gốc (Orchestrator) luôn giữ được sự trong sạch, context của nó chỉ toàn là "lệnh người dùng" và "kết quả tóm gọn", không bao giờ gặp tình trạng kiệt sức giới hạn lưu trữ tokens.
