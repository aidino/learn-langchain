# DeepAgents: Async Subagents (Tiểu tác tử Bất đồng bộ)

Tài liệu này giải thích cách sử dụng **Async Subagents** trong DeepAgents, cho phép bạn khởi chạy các tiểu tác tử chạy ngầm (background) một cách đồng thời, trong khi tác tử giám sát (supervisor) vẫn có thể tiếp tục tương tác với người dùng.

## 1. Khi nào nên sử dụng Async Subagents?

Async subagents xử lý các tác vụ mất nhiều thời gian mà không làm chặn (block) luồng xử lý của tác tử chính. Chúng rất hữu ích trong các trường hợp:
- Cần chạy các tác vụ chạy ngầm cần nhiều thời gian (như nghiên cứu chuyên sâu, phân tích dữ liệu lớn).
- Cần khởi chạy nhiều tác vụ cùng một lúc và thu thập kết quả sau.
- Tận dụng các máy chủ hoặc dịch vụ deployment từ xa chuyên biệt.

Khi được cấu hình, Tác tử sẽ được cung cấp các công cụ như `start_async_task`, `check_async_task`, `update_async_task`, `cancel_async_task` và `list_async_tasks` để có thể chủ động điều phối và quản lý các agent con này.

## 2. Cấu hình Async Subagents

Bạn định nghĩa các Async Subagents thông qua lớp `AsyncSubAgent` hoặc dưới dạng `dict`, sau đó truyền vào hàm `create_deep_agent`.

```python
from deepagents import AsyncSubAgent, create_deep_agent

async_subagents = [
    AsyncSubAgent(
        name="researcher",
        description="Tiểu tác tử nghiên cứu để thu thập và tổng hợp thông tin",
        graph_id="researcher", 
        # Không truyền url -> sử dụng ASGI transport (chạy cùng một máy chủ deployment)
    ),
    AsyncSubAgent(
        name="coder",
        description="Tiểu tác tử lập trình để tạo và review code",
        graph_id="coder",
        url="https://coder-deployment.langsmith.dev" # VÍ DỤ: Sử dụng HTTP transport để gọi máy chủ từ xa
    ),
]

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    subagents=async_subagents,
)
```

**Các tham số chính của cấu hình Subagent:**
- `name`: Tên của tiểu tác tử (được cung cấp cho LLM dưới dạng một bộ tool gọi AI).
- `description`: Mô tả chi tiết cho công cụ. Điều này rất quan trọng để LLM biết chính xác khi nào nên phân chia công việc cho subagent.
- `graph_id`: ID của graph được định nghĩa trong tệp `langgraph.json` ở phía máy chủ.
- `url`: (Tuỳ chọn) URL của máy chủ triển khai subagent. Nếu bỏ trống, framework sẽ hiểu là subagent được cấu hình cùng nội bộ dự án.
- `headers`: (Tuỳ chọn) Chứa các headers dưới dạng dictionary (ví dụ: API Key để ủy quyền).

## 3. Best Practices cho DeepAgents (Phiên bản `0.5.3`)

Dựa trên tìm kiếm tài liệu từ thư viện `context7`, dưới đây là ví dụ tốt nhất (best practice) để tích hợp một Async Subagent gọi đến một remote server cho phiên bản 0.5.3:

```python
from deepagents import create_deep_agent, AsyncSubAgent

# Cấu hình async subagent trỏ đến một remote deployment
remote_researcher: AsyncSubAgent = {
    "name": "deep-researcher",
    "description": "Thực hiện các tác vụ nghiên cứu chuyên sâu dưới dạng chạy ngầm trên một máy chủ chuyên dụng. Sử dụng cho các câu hỏi cần search nhiều bước kết hợp.",
    "graph_id": "research-agent",  # Tên Graph nằm trên máy chủ từ xa
    "url": "https://my-langgraph-deployment.langchain.app",  # URL triển khai của subagent
}

# Tạo supervisor agent bao gồm async subagent
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    subagents=[remote_researcher],
)

# Kích hoạt lệnh, tác tử sẽ dùng tool và trả ngay về Task ID
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Hãy khởi chạy một tác vụ nghiên cứu chuyên sâu về xu hướng điện toán lượng tử"}
    ]
})
```

**Những lưu ý quan trọng về thiết kế (Best Practices):**
1. **Viết mô tả (description) thật rõ ràng và giới hạn phạm vi:** Đừng viết chung chung (ví dụ: "giúp tôi làm việc"). Hãy trình bày cụ thể về input và output (ví dụ: "Dùng để nghiên cứu thông qua web. Yêu cầu một câu hỏi tổng hợp lớn").
2. **Ngăn tác tử chính poll kết quả ngay lập tức:** Ở bước thiết lập `system_prompt` cho tác tử supervisor, hãy cấu hình cụ thể rằng: *"Sau khi khởi chạy một subagent bất đồng bộ, hãy báo cáo ID và trả về kết quả cho người dùng. KHÔNG ĐƯỢC gọi `check_async_task` ngay lập tức."*
3. **Trace log bằng Thread IDs:** `task_id` được trả về cho Supervisor khi khởi động tác vụ bất đồng bộ thực chất chính là thread ID trên LangGraph Server điều phối. Hãy dùng nó nếu bạn cần theo dõi log hệ thống.

## 4. Vòng đời (Lifecycle) và Quản lý Trạng thái (State Management)

Dưới lớp middleware của hệ thống (`AsyncSubAgentMiddleware`), Tác tử (Supervisor) sẽ tương tác với Subagent qua vòng đời 5 bước tiêu chuẩn:

1. **Launch (Bắt đầu):** Supervisor yêu cầu tạo một Thread mới trên hệ thống, bắt đầu chạy một luồng độc lập với mô tả đầu vào. Supervisor nhận lại Thread ID (đóng vai trò Task ID) và **lập tức quay lại phục vụ người dùng**, không bị blocking.
2. **Check (Kiểm tra):** Báo cáo trạng thái hiện tại của Thread (như `success`, `error`, hoặc `cancelled`). Nếu `success`, nó lấy trực tiếp output cuối cùng ra.
3. **Update (Cập nhật):** Supervisor có thể yêu cầu khởi tạo run mới trên cùng thread ID bằng chiến lược ngắt quãng (`interrupt multitask`). Subagent sẽ hủy quá trình hiện tại và tự động khởi động lại với toàn bộ lịch sử (context) hiện có kết hợp chỉ thị mới.
4. **Cancel (Huỷ):** Supervisor gọi API điều khiển hệ thống (`.cancel()`) để dừng task.
5. **List (Liệt kê):** Truy xuất live qua các Task đã lưu và trả về danh sách lịch trình.

_Quản lý trạng thái_ của Supervisor phụ thuộc vào việc hệ thống ghi nhớ các Task ID đang mở. Nếu dự án của bạn sử dụng chiến lược rút gọn bộ nhớ dữ liệu (Summarization/Trimming đối với Context Engineering), hãy chắc chắn phải chừa lại danh sách `async_tasks` trong State gốc của LLM để agent chính không bị "quên" các tiến trình ngầm trị giá cao đang chạy.

## 5. Chọn Phương thức kết nối (Transport) và Cấu trúc Triển khai (Topology)

DeepAgents cho phép 2 loại Transport chính:
*   **ASGI Transport (Triển khai cùng một khối):** Khi bạn không chỉ định tham số `url`, hệ thống sẽ cho rằng cả supervisor và subagent đang cùng cấu trúc dự án. LangChain sẽ duyệt file `langgraph.json` trên ứng dụng để gọi nội bộ hàm xử lý.
*   **HTTP Transport (Triển khai Rời/Từ xa):** Khi có tham số `url`, hệ thống chuyển từ gọi module sang gọi HTTP client theo giao thức Agent Client Protocol (ACP). Sẽ cần cấu hình xác thực như `LANGSMITH_API_KEY` nếu bạn deploy lên hệ sinh thái đám mây của LangGraph.

Các kiểu Topology thường lập trình:
1.  **Single Deployment (Quy mô nhỏ):** Mọi tác tử cùng sống trên một application codebase duy nhất.
2.  **Split Deployment (Services Độc Lập):** Supervisor và Subagent nằm rải rác trên các service hoặc repository khác nhau. Tối ưu theo cấu hình máy chủ riêng biệt (VD: Agent phân tích số liệu dùng cụm có nhiều RAM).
3.  **Hybrid (Lai):** Tác tử cơ bản chạy internal, tác tử chuyên biệt chạy external.

## 6. Khắc phục sự cố (Troubleshooting)

- **Lỗi 1: Supervisor liên tục kiểm tra lỗi ngay sau khi Launch.** (Lặp vô hạn vòng loop).
  * **Giải pháp**: Xây dựng system prompt chặt chẽ, bắt buộc LLM chỉ trả về `task_id` và tương tác kết nối người dùng thay vì gọi chức năng chờ (`check_async_task`).
- **Lỗi 2: Trạng thái trả về có vẻ sai (Stale status)/LLM bịa ra kết quả.** 
  * **Hiện tượng**: LLM ảo giác tự nhớ lại chuỗi câu chuyện đã chạy thay vì gọi tool check thực tế.
  * **Giải pháp**: Khuyến khích mạnh mẽ trong prompt cho LLM phải dùng list tool hoặc check tool để cập nhật logic trạng thái thực sự vào pipeline.
- **Lỗi 3: Subagent nằm trong hàng đợi (Queued) ở local chờ một thời gian rất dài.**
  * **Giải pháp**: Xảy ra ở môi trường local development vì mặc định giới hạn số task worker quá nhỏ. Hãy khởi chạy project local bằng lệnh `langgraph dev --n-jobs-per-worker 10` để tránh tắc nghẽn queue.
