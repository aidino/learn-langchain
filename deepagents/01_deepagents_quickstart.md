# Hướng dẫn Bắt đầu Nhanh với Deep Agents (LangChain)

Hướng dẫn này cung cấp các bước cơ bản để xây dựng "Deep Agents" sử dụng LangChain, tập trung vào việc tạo ra một agent nghiên cứu có khả năng tự động lập kế hoạch và tạo ra các agent phụ (subagent).

## Tổng quan (Overview)
**Deep Agents** được thiết kế để xử lý các tác vụ phức tạp bằng cách:
- Tự động lập kế hoạch (Planning).
- Sử dụng các công cụ (như hệ thống tệp tin, công cụ tìm kiếm).
- Sinh ra (spawn) các subagent để xử lý các nhiệm vụ nhỏ hơn hoặc chuyên biệt.

---

## Các bước thực hiện

### Bước 1: Cài đặt thư viện phụ thuộc (Dependencies)
Bạn cần cài đặt thư viện `deepagents` và một nhà cung cấp dịch vụ tìm kiếm (ở ví dụ này sử dụng Tavily).
```bash
pip install deepagents tavily-python
```

### Bước 2: Thiết lập API Keys
Bạn cần cấu hình API key cho mô hình ngôn ngữ (ví dụ: Anthropic) và công cụ tìm kiếm (Tavily).
```bash
export ANTHROPIC_API_KEY="your-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

### Bước 3: Tạo công cụ Tìm kiếm (Search Tool)
Định nghĩa một hàm để agent có thể sử dụng nhằm thu thập thông tin từ internet. Ở đây, chúng ta tạo hàm `internet_search` sử dụng `TavilyClient`.
```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

# Khởi tạo client cho Tavily
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Thực hiện tìm kiếm trên web"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
```

### Bước 4: Khởi tạo Deep Agent
Khởi tạo agent với một mô hình AI, công cụ tìm kiếm vừa tạo, và các hướng dẫn (system prompt).
```python
# Hướng dẫn (Prompt) hệ thống cho agent
research_instructions = """Bạn là một chuyên gia nghiên cứu. Công việc của bạn là tiến hành nghiên cứu kỹ lưỡng và sau đó viết một báo cáo hoàn chỉnh.
Bạn có quyền truy cập vào công cụ tìm kiếm trên internet như là phương tiện chính để thu thập thông tin."""

# Tạo agent
agent = create_deep_agent(
    model="anthropic:claude-3-5-sonnet-latest", # Chuỗi định danh mô hình (ví dụ dùng Claude 3.5 Sonnet)
    tools=[internet_search],
    system_prompt=research_instructions,
)
```

### Bước 5: Chạy Agent
Kích hoạt (invoke) agent với một câu hỏi từ người dùng và in ra kết quả.
```python
# Gọi agent với câu hỏi cụ thể, ví dụ: "langgraph là gì?"
result = agent.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})

# In ra câu trả lời cuối cùng
print(result["messages"][-1].content)
```

---

## Cơ chế hoạt động (How It Works)
Khi một Deep Agent được thực thi, nó sẽ tự động thực hiện quy trình sau:
1. **Lập kế hoạch (Planning):** Sử dụng công cụ `write_todos` tích hợp sẵn để chia nhỏ tác vụ thành các bước cần làm.
2. **Nghiên cứu (Research):** Gọi công cụ `internet_search` (hoặc các công cụ khác được cung cấp) để thu thập dữ liệu.
3. **Quản lý Ngữ cảnh (Context Management):** Sử dụng các công cụ hệ thống tệp (như `write_file`, `read_file`) để quản lý lượng lớn dữ liệu mà không bị quá tải bộ nhớ ngữ cảnh (context window) của mô hình.
4. **Ủy quyền (Delegation):** Tạo ra các subagent chuyên biệt để giải quyết các tác vụ con phức tạp, giúp chia để trị hiệu quả.
5. **Tổng hợp (Synthesis):** Tổng hợp tất cả các thông tin tìm được thành một báo cáo hoặc câu trả lời cuối cùng.

## Các Tính năng Chính (Key Features)
* **Streaming (Phát trực tuyến):** Hỗ trợ cập nhật theo thời gian thực các lần gọi công cụ và luồng phản hồi của LLM thông qua sức mạnh của LangGraph.
* **Human-in-the-loop (Sự tham gia của con người):** Có thể được cấu hình để yêu cầu sự phê duyệt hoặc can thiệp thủ công từ người dùng trước khi tiến hành các bước quan trọng (ví dụ: xác nhận kế hoạch, cho phép thực thi công cụ).
* **Memory (Bộ nhớ lưu trữ):** Hỗ trợ bộ nhớ lưu trữ bền vững (persistent memory) giúp agent có thể nhớ thông tin qua các phiên hội thoại khác nhau.
