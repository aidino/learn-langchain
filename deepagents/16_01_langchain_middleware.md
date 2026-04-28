# Hướng dẫn chi tiết về LangChain Middleware và DeepAgents 0.5.3

Tài liệu này dịch và giải thích các khái niệm về Middleware trong hệ sinh thái LangChain, đồng thời bổ sung các best practices thực tế khi làm việc với thư viện `deepagents==0.5.3`.

---

## 1. Tổng quan về Middleware (Overview)
Middleware trong LangChain cung cấp khả năng kiểm soát và tùy biến quá trình thực thi của Agent tại mỗi bước nhỏ nhất. Việc sử dụng middleware mang lại các lợi ích:
- **Theo dõi & Debug:** Ghi log, phân tích hành vi của agent.
- **Biến đổi Dữ liệu:** Sửa đổi prompt, tùy chỉnh lựa chọn tool (tool selection), định dạng lại output.
- **Độ tin cậy:** Thêm cơ chế retry (thử lại), fallback (chuyển đổi model dự phòng), và logic dừng sớm.
- **An toàn & Chi phí:** Áp dụng rate limits (chặn vượt quá số lần gọi api), guardrails, phát hiện và che giấu thông tin cá nhân (PII).

Hầu hết middleware được truyền thông qua tham số `middleware` khi khởi tạo agent thông thường bằng `create_agent()`.

---

## 2. Prebuilt Middleware (Middleware tích hợp sẵn)
LangChain cung cấp đa dạng các middleware đã được xây dựng sẵn để giải quyết các bài toán phổ biến:

### Quản lý Ngữ Cảnh & Ngân sách
- **`SummarizationMiddleware`**: Khi lịch sử hội thoại quá dài vượt ngưỡng kích hoạt (ví dụ: `tokens >= 4000` hoặc `fraction == 0.8`), middleware này sẽ sử dụng một model nhỏ để tóm tắt các tin nhắn cũ, giúp tiết kiệm bộ nhớ và token.
- **`ModelCallLimitMiddleware` & `ToolCallLimitMiddleware`**: Ngăn chặn tình trạng agent rơi vào vòng lặp vô tận (runaway) hoặc gọi các API tính phí quá mức. Ta có thể định cấu hình `exit_behavior` (ví dụ: `error` để văng lỗi hoặc `continue` để giữ lại luồng).
- **`LLMToolSelectorMiddleware`**: Trường hợp bạn có quá nhiều tools (ví dụ >10), middleware này dùng LLM rẻ để lọc trước ra các tools cần thiết trước khi đẩy cho model chính, tiết kiệm chi phí và giúp model tập trung.

### An toàn & Tự động hoá
- **`HumanInTheLoopMiddleware`**: Dành cho thao tác rủi ro cao (chuyển tiền, gửi email). Yêu cầu phê duyệt từ con người qua `approve`, `reject`, hay `edit`.
- **`PIIMiddleware`**: Tự động chặn, mã hóa (mask), hoặc viết lại các thông tin nhận dạng cá nhân (PII) như email, số tín dụng trả về từ AI. Tính năng hỗ trợ chạy qua các regex tùy thích hoặc hàm Python.
- **`TodoListMiddleware`**: Giúp Agent lưu một todo-list ảo để theo sát các luồng công việc nhiều bước (complex multi-step tasks) tốn kém nhiều thời gian theo dõi.

### Khả năng phục hồi (Resilience)
- **`ModelFallbackMiddleware`**: Nếu model chính bị lỗi mạng hay quá tải, middleware sẽ chuyển câu lệnh tự động sang model dự phòng nhằm đảm bảo trải nghiệm trôi chảy.

---

## 3. Custom Middleware (Tự tạo Middleware tuỳ chỉnh)
Nếu middleware có sẵn không đủ đáp ứng, bạn có thể tạo Custom Middleware dưới dạng **Hàm decorator** hoặc **Class**.

### Các loại Hooks
- **Node-style hooks** (`@before_agent`, `@after_agent`, `@before_model`, `@after_model`): Sửa trực tiếp dictionary trả về vào agent state dựa trên logic tuần tự. Phù hợp cho việc logging, kiểm soát độ dài messages, validation.
- **Wrap-style hooks** (`@wrap_model_call`, `@wrap_tool_call`): Bọc (wrap) toàn bộ một model call hoặc tool call. Thường dùng khi cần kiểm soát luồng điều khiển, chẳng hạn như lặp lại gọi API (retry) khi gặp lỗi hoặc caching.

### Quản lý trạng thái nội bộ (Custom State Schema)
Bạn có thể khai báo class kế thừa `AgentState` để theo dõi giá trị xuyên suốt luồng. Ví dụ như đếm số tokens, lưu `user_id`. Middleware sẽ nhận biến này và cập nhật qua reducer (hàm quản lý cộng gộp state của LangGraph).

### Thứ tự thực thi
- `before_*`: Thực thi theo thứ tự danh sách (Từ trên xuống).
- `after_*`: Thực thi theo thứ tự ngược (Từ dưới lên).
- `wrap_*`: Lồng nhau, middleware nằm trên sẽ bọc middleware nằm dưới. Hỗ trợ thao tác cập nhật bằng hàm `Command`.

---

## 4. DeepAgents (0.5.3) - Best Practices & Tích hợp
Với thư viện `deepagents==0.5.3`, kiến trúc middleware được tích hợp sâu vào việc thiết lập ngữ cảnh động. Bạn không cần phải khai báo thủ công nhiều Middleware cơ bản vì hàm kiến trúc `create_deep_agent` đã bao bọc sẵn `TodoList`, `Filesystem`, và `SubAgent` middleware.

### Ví dụ Thực tiễn (Best Practice Example)
Dưới đây là phương pháp khởi tạo chuẩn cho `deepagents==0.5.3` với cấu hình nạp middleware tối ưu, kết hợp Sandboxing thông qua FilesystemBackend nhằm cô lập môi trường thực thi:

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.tools import web_search
# Load subagents từ một config YAML bên ngoài giúp code gọn gàng hơn
from deepagents.utils import load_subagents 

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    
    # Tính năng tích hợp của DeepAgents 0.5.3:
    # Truyền trực tiếp đường dẫn file, 'MemoryMiddleware' sẽ tự động dịch và nhúng 
    # file này vào system prompt dưới dạng bối cảnh dài hạn, thông qua class chặn modify_request().
    memory=["/workspace/AGENTS.md", "/memories/project-context.md"],
    
    # 'SkillsMiddleware' sẽ nạp kỹ năng phụ trợ động từ thư mục, gán vào lúc gọi tool
    skills=["./skills/"],
    
    tools=[web_search],
    
    # Subagent bắt buộc định nghĩa trong script hoặc load qua YAML thay vì inject tự động như string
    subagents=load_subagents("./subagents.yaml"),
    
    # Cách ly môi trường thao tác tệp tin giúp an toàn tuyệt đối khi thực thi Code
    backend=FilesystemBackend(root_dir="/workspace/secure_dir/"),
)

# Bạn cũng có thể tiêm nội dung files ngay trong luồng invoke qua cơ chế state mặc định
result = agent.invoke({
    "messages": [{"role": "user", "content": "Nguyên tắc lập trình hiện tại của dự án là gì?"}],
    "files": {
        "/workspace/AGENTS.md": {
            "content": "# Nguyên tắc\n\n- Tuân thủ REST API strictly\n- Sử dụng TypeScript cho frontend",
            "encoding": "utf-8"
        }
    }
})
```

### Ghi chú quan trọng về Bảo mật (Threat Model)
- Trong `deepagents 0.5.3`, quá trình nạp bằng `MemoryMiddleware` hay `SkillsMiddleware` (thông qua hàm `.modify_request()`) sẽ nội suy và chèn trực tiếp nội dung các files vào hệ thống qua hàm `str.format()`. 
- Framework mặc định **zero validation** đối với phần nội dung này (User/Framework trust boundary). Ngữ cảnh được đưa thẳng vào hệ thống, do đó bạn cần hết sức lưu ý các đường dẫn thư mục `memory` hay `skills` đến từ user input có bị nhiễm mã độc (Prompt Injection) hay không; cũng như kiểm soát backend truy cập file cho đúng nguyên tắc.
