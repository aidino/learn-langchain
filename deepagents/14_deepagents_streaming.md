# Streaming trong DeepAgents

DeepAgents được xây dựng dựa trên cơ sở hạ tầng streaming của LangGraph, cung cấp first-class support (tính năng cốt lõi) để truyền các subagent streams (luồng dữ liệu thời gian thực từ các subagent). Khi một agent chính ủy quyền công việc cho các subagent, bạn có thể truyền tải các cập nhật từ từng subagent một cách độc lập—cho phép theo dõi tiến độ, LLM tokens, và các lệnh gọi tool ngay tại thời điểm thực thi.

**Những khả năng có thể thực hiện với deep agent streaming:**
- **Stream tiến trình subagent:** theo dõi quá trình chạy của từng subagent song song.
- **Stream LLM tokens:** nhận từng token văn bản từ main agent và bất kỳ subagent nào đang hoạt động.
- **Stream tool calls:** xem việc gọi công cụ và kết quả từ trong quá trình chạy của subagent.
- **Stream cập nhật tùy chỉnh (custom updates):** phát ra các tín hiệu tiến trình theo ý bạn từ bên trong các công cụ.

---

## Kích hoạt Subgraph Streaming

DeepAgents sử dụng chế độ "subgraph streaming" của LangGraph để hiển thị các event từ subagent. Để nhận các event từ subagent, bạn cần bật tham số `subgraphs=True` và khai báo version v2 (`version="v2"`) khi gọi hàm stream.

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    system_prompt="You are a helpful research assistant",
    subagents=[
        {
            "name": "researcher",
            "description": "Researches a topic in depth",
            "system_prompt": "You are a thorough researcher.",
        },
    ],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Research quantum computing advances"}]},
    stream_mode="updates",
    subgraphs=True,  # Bật stream cho subgraphs [!code highlight]
    version="v2",    # Sử dụng định dạng streaming v2 [!code highlight]
):
    if chunk["type"] == "updates":
        if chunk["ns"]:
            # Sự kiện Subagent - namespace xác định nguồn phát ra
            print(f"[subagent: {chunk['ns']}]")
        else:
            # Sự kiện từ Main agent
            print("[main agent]")
        print(chunk["data"])
```

## Namespaces (Không gian tên)

Khi tham số `subgraphs` được kích hoạt, mỗi sự kiện streaming sẽ đi kèm một trường `namespace` (`ns`) để giúp bạn xác định agent nào đã tạo ra sự kiện đó. Namespace là một dãy/đường dẫn các tên node và task ID biểu hiện thứ bậc của agent.

- `()` (Rỗng): Sự kiện do Main agent phát ra.
- `("tools:abc123",)`: Sự kiện từ một subagent được phóng/sinh ra từ lệnh gọi `task` của main agent có id là `abc123`.
- `("tools:abc123", "model_request:def456")`: Gọi node `model_request` ở sâu bên trong một subagent.

Gợi ý cấu trúc logic để định tuyến UI với Namespaces:
```python
is_subagent = any(segment.startswith("tools:") for segment in chunk["ns"])
if is_subagent:
    # Lấy ID của lệnh gọi tool (chính là ID của subagent task)
    tool_call_id = next(s.split(":")[1] for s in chunk["ns"] if s.startswith("tools:"))
    print(f"Subagent {tool_call_id}: {chunk['data']}")
else:
    print(f"Main agent: {chunk['data']}")
```

---

## 1. Stream Tiến Độ Subagent (`updates`)

Sử dụng `stream_mode="updates"` để theo dõi tiến độ của subagent khi mỗi bước (node) hoàn tất. Việc này cực kỳ hữu ích để hiển thị subagent nào đang bận và khối lượng công việc nào vừa hoàn thành.

```python
# Khi nhận kết quả từ chunk["data"] (dạng dictionary chứa giá trị cập nhật state):
for chunk in agent.stream(..., stream_mode="updates", subgraphs=True, version="v2"):
    if chunk["type"] == "updates":
        if not chunk["ns"]: # Main Agent
            for node_name, data in chunk["data"].items():
                if node_name == "tools":
                    # Xử lý kết quả subagent trả về main agent
                    pass 
                else:
                    print(f"[main agent] step: {node_name}")
        else:
            for node_name, data in chunk["data"].items():
                print(f"  [{chunk['ns'][0]}] step: {node_name}")
```

## 2. Stream LLM Tokens (`messages`)

Sử dụng `stream_mode="messages"` để nhận token ngay lúc LLM sinh ra. Điều này áp dụng cho cả main agent và subagents.

Mỗi chunk dạng `messages` sẽ phân tách được nội dung token ở `chunk["data"]`:
```python
for chunk in agent.stream(..., stream_mode="messages", subgraphs=True, version="v2"):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        
        is_subagent = any(s.startswith("tools:") for s in chunk["ns"])
        if is_subagent:
            print(token.content, end="", flush=True)
        else:
            # Token từ Main agent
            print(token.content, end="", flush=True)
```

## 3. Custom Updates (Cập nhật tùy chỉnh)

Đôi khi bạn muốn truyền một tín hiệu phân tích theo ý định của mình (vd: cập nhật thanh % tiến trình) khi chạy một công cụ nặng nề. Bạn có thể sử dụng hàm `get_stream_writer()` của LangGraph bên trong tool.

```python
import time
from langchain.tools import tool
from langgraph.config import get_stream_writer

@tool
def analyze_data(topic: str) -> str:
    """Run data analysis."""
    writer = get_stream_writer()
    
    # Phát ra các sự kiện progress (khi nhận ở chế độ "custom")
    writer({"status": "starting", "topic": topic, "progress": 0})
    time.sleep(0.5)
    writer({"status": "analyzing", "progress": 50})
    time.sleep(0.5)
    writer({"status": "complete", "progress": 100})
    
    return "Analysis complete."

# Mode: stream_mode="custom"
# if chunk["type"] == "custom": 
#    print(chunk["data"])  -> In ra {"status": "starting"...}
```

## 4. Theo Dõi Vòng Đời Subagent

Bạn có thể theo dõi lifecycle (vòng đời) trạng thái của subagent bằng cách đọc mode `updates`:
1. **PENDING (Đang chờ):** Bắt được khi `node_name == "model_request"`, Main agent tạo list `tool_calls` cho object `task`.
2. **RUNNING (Đang chạy):** Bắt được khi nhận event đầu tiên từ namespace có format `tools:ID`.
3. **COMPLETE (Hoàn tất):** Bắt được khi `node_name == "tools"` (thuộc main namespace rỗng) có trả về tool message trả lời cho `task`.

## 5. Định dạng Streaming V2

DeepAgents khuyến khích sử dụng **v2 streaming format** (`version="v2"`), yêu cầu từ `LangGraph >= 1.1`. Toàn bộ dữ liệu stream ở chuẩn V2 là một Dict duy nhất có luôn các cột báo hiệu, giúp loại bỏ việc bóc tách `nested tuples` như phiên bản trước kia.

```python
# CHUẨN V2 MỚI
for chunk in agent.stream(..., version="v2"):
    print(chunk["type"])  # "updates", "messages", "custom"...
    print(chunk["ns"])    # () cho main agent, ("tools:<abc>",) cho subagent
    print(chunk["data"])  # Dữ liệu payload
```

---

## 🌟 Best Practices: DeepAgents 0.5.3 (Xử lý Context & File Data Stream lớn)

Dưới đây là một số **Thực hành tốt nhất (Best Practices)** khi sử dụng DeepAgents phiên bản 0.5.3 từ thư viện `langchain-ai/deepagents`.

### Xử lý Tải File Lớn và Giao Thức Mạng kết hợp Subagents

1. **Nguyên tắc Ủy quyền File (Delegate Pattern):** Đừng cố nhồi Data vào Token LLM thông qua Streaming Tools qua Context Window. **Hãy tải bất kỳ tài liệu hay file từ mạng xuống filesystem trước**, sau đó kích hoạt subagent (`data-processor-agent`) phân tích tài liệu và để subagent tự do xem cấp đọc tài liệu qua file.
   - **Vì sao?** Bởi vì DeepAgent framework cho phép Subagent chia sẻ chung một hệ thống Filesystem với Main Agent, tải và lưu vào Disk là giải pháp tối ưu nhất cho hiệu năng.

2. **Stream URL Data Hiệu Quả với Dừng Sớm (Early Termination) trong các Plugin:**
   Nếu bạn viết custom code/tool tải file qua HTTP, đừng load toàn bộ file vào RAM khi chỉ cần N rows, sử dụng kết hợp `stream=True` và dừng tải qua ngắt `iter_lines()` sớm:
   ```python
   import requests, os
   os.makedirs('/data', exist_ok=True)
   url = "https://example.com/large-data.csv"
   
   # Mô hình tải siêu tiết kiệm bộ nhớ cho DeepAgents 0.5.3
   with requests.get(url, stream=True, timeout=30) as r:
       r.raise_for_status()
       with open('/data/output.csv', 'w') as f:
           count = 0
           for line in r.iter_lines(decode_unicode=True):
               if line:
                   f.write(line + '\n')
                   count += 1
                   if count >= 1001:  # Lưu 1 header + 1000 dòng data
                       break          # Ngắt Streaming sớm để dọn RAM
   ```
   *Quá trình này hoạt động vô cùng đáng tin cậy. Kết nối mạng được hủy ngay khi ta có đủ số dòng, không tạo ra áp lực về bộ nhớ hệ thống (memory pressure) và SubAgent có thể ngay lập tức parse csv từ ổ đĩa.*

3. **Xử lý Sever Bị Treo (Fallback for Stalled Connections):** Nếu Remote server bỏ qua tín hiệu cớ ngắt Streaming sớm bằng HTTP thường dẫn đến block/treo Tool, **hãy sử dụng Socket SSL Cấp Thấp với giao thức HTTP/1.0**. Giao thức này không hỗ trợ Chunked transfer, nhận đủ dữ liệu trực tiếp đóng socket, ép Force-Close hiệu quả cho độ ổn định Agent ở tầng Production.
