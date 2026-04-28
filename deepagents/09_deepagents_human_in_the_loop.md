# DeepAgents: Human-in-the-loop (Vòng lặp có sự tham gia của con người)

Tài liệu này hướng dẫn cách thiết lập sự phê duyệt và can thiệp từ con người trước khi Agent thực hiện các lệnh nhạy cảm (như ghi file, xóa file, thực thi mã shell...).

## Tổng quan (Overview)
Trong DeepAgents, **Human-in-the-loop (HITL)** là cơ chế cho phép hệ thống tạm dừng (interrupt) quá trình thực thi để xin ý kiến từ người dùng thật. Cơ chế này đặc biệt quan trọng để:
- **Kiểm soát rủi ro:** Đảm bảo các công cụ (tools) có tính phá hủy hoặc rủi ro cao (chẳng hạn như xóa tài nguyên, chạy mã hệ thống, gửi email cho khách hàng) phải được xác nhận.
- **Bảo mật và chặn Prompt Injection:** Bằng cách buộc hiển thị tham số công cụ và chờ duyệt, bạn có thể ngăn chặn Agent khỏi việc tự động thực thi các hành động trái phép do tấn công Prompt Injection theo cách thức an toàn nhất.

---

## 1. Cấu hình cơ bản (Basic configuration)

Tham số `interrupt_on` quản lý việc bạn muốn dừng agent ở những công cụ nào.  
Có 3 giá trị thiết lập cho mỗi công cụ:
- `True`: Kích hoạt chế độ ngắt ngầm định (người dùng có quyền **phê duyệt, chỉnh sửa tham số, từ chối**).
- `False`: Vô hiệu hóa ngắt (Agent tự do dùng công cụ này).
- `{"allowed_decisions": [...]}`: Cấu hình tuỳ chỉnh, giới hạn các quyết định mà người dùng được phép đưa ra.

```python
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

@tool
def delete_file(path: str) -> str:
    """Xóa file khỏi hệ thống."""
    return f"Deleted {path}"

@tool
def read_file(path: str) -> str:
    """Đọc nội dung file."""
    return f"Contents of {path}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Gửi email."""
    return f"Sent email to {to}"

# LƯU Ý: Rất quan trọng, bắt buộc phải có Checkpointer khi dùng human-in-the-loop
checkpointer = MemorySaver()

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    tools=[delete_file, read_file, send_email],
    interrupt_on={
        "delete_file": True,  # Mặc định: cho phép phê duyệt, chỉnh sửa, từ chối
        "read_file": False,   # An toàn, không cần chờ người dùng xác nhận
        "send_email": {"allowed_decisions": ["approve", "reject"]}, # Chỉ được duyệt/từ chối, không thể sửa nội dung email
    },
    checkpointer=checkpointer # Bắt buộc phải truyền checkpointer
)
```

---

## 2. Các loại quyết định (Decision types)
Bạn có thể cấu hình tham số tuỳ chỉnh qua `allowed_decisions`. Các loại quyết định bao gồm:
- `"approve"` (Phê duyệt): Thực thi công cụ với đúng tham số mà Agent đề xuất.
- `"edit"` (Chỉnh sửa): Cho phép ứng dụng/giao diện người dùng can thiệp sửa đổi các tham số của công cụ trước khi thực thi.
- `"reject"` (Từ chối): Từ chối thực thi và bỏ qua lời gọi công cụ đó (Agent sẽ được thông báo rằng công cụ bị lỗi hoặc bị từ chối).

---

## 3. Xử lý sự kiện Ngắt (Handle interrupts)
Khi gọi agent, nếu agent chạm đến tool cần xác nhận, tiến trình sẽ tạm dừng. Bạn cần kiểm tra cờ (flag) `interrupts` trên kết quả trả về, giải nén dữ liệu, và dùng `Command(resume=...)` để trả lời agent.

```python
from langchain_core.utils.uuid import uuid7
from langgraph.types import Command

# Bắt buộc khai báo thread_id trong config để lưu trạng thái phiên làm việc
thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Xóa file temp.txt"}]},
    config=config,
    version="v2",
)

# Nếu kết quả có chứa `interrupts`, hệ thống đang chờ thao tác từ con người
if result.interrupts:
    interrupt_value = result.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    
    # In ra công cụ đang chờ duyệt để người dùng xem
    for action in action_requests:
        print(f"Agent muốn gọi tool: {action['name']} với tham số {action['args']}")

    # Giả lập thao tác người dùng: Phê duyệt
    decisions = [
        {"type": "approve"} 
    ]

    # Muốn chỉnh sửa (Edit) có thể gửi như sau:
    # decisions = [{
    #     "type": "edit",
    #     "edited_action": {
    #         "name": action["name"],
    #         "args": {"path": "correct_file.txt"}
    #     }
    # }]

    # Nối lại luồng với câu trả lời (phải dùng chung cấu hình config + thread_id)
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
        version="v2",
    )
```

> **Lưu ý thao tác đa lệnh (Multiple tool calls)**: Nếu Agent gọi cùng lúc 2 công cụ cần xin phép, danh sách mảng `decisions` lúc `resume` của bạn **phải khớp đúng thứ tự** với mảng `action_requests`.

---

## 4. Ngắt luồng ở cấp độ Subagents
Bên cạnh Agent chính, DeepAgents còn hỗ trợ quản lý HITL tinh tế ở cấp độ **Subagent** và cấp độ **bên trong Tool**.

* **Ngắt qua Tool Calls**: Bạn có thể định nghĩa tham số `interrupt_on` riêng biệt khi khai báo Subagent, nhằm đè (override) cấu hình của Agent chính. Việc này cho phép bạn xây dựng cấu hình duyệt rủi ro theo từng agent con chuyên biệt.
* **Ngắt từ bên trong ruột Tool**: Bạn có thể tự gọi hàm nguyên thuỷ `interrupt({...})` của kiến trúc `langgraph` ngay bên trong function logic của `@tool`. Kết quả trả về của hàm này sẽ tạm dừng tiến trình thực thi của python ngay lập tức.

---

## 5. Thực hành tốt nhất (Best Practices) cập nhật cho v0.5.3

Dựa trên tìm kiếm và tư vấn kiến trúc của thư viện `deepagents==0.5.3` (từ Context7/LangGraph):

1. **Luôn sử dụng một Checkpointer (Bắt buộc)**
    Luồng HITL thay đổi cấu trúc đồ thị luồng xử lý và cần cơ chế lưu và tạm dừng. Bạn bắt buộc phải truyền `MemorySaver()` (hoặc Checkpointer tương tự) vào biến `checkpointer` của `create_deep_agent`.
   
2. **Duy trì ổn định Thread ID (`use the same thread ID`)**
    Bất kỳ lúc nào gửi dữ liệu qua `Command(resume=...`, phải đảm bảo rằng biến `config={"configurable": {"thread_id": "thread-cu-cua-ban"}}` được sử dụng hoàn toàn giống với lúc `invoke` ban đầu để hệ thống nạp lại bộ nhớ.

3. **Chặn lệnh Execute và bảo vệ Prompt Injection**
    Trong môi trường ứng dụng DeepAgents cho phép thao tác hệ thống, công cụ rủi ro cao nhất là `execute` (thực thi shell code). Khi khởi tạo agent, bạn nên tuỳ chỉnh quyền hạn theo mức độ công việc:
    ```python
    interrupt_on={
        "write_file": True,            # Dừng để đảm bảo nội dung ghi là an toàn
        "edit_file": True,             # Dừng trước khi ghi đè file có sẵn
        "execute": {"always": True},   # TUYỆT ĐỐI luôn luôn dừng shell command 
        "read_file": False,            # An toàn đọc
    }
    ```
    Đây là biện pháp hiệu quả mạnh mẽ cho Threat Model của DeepAgents nhắm vào khai thác lệnh (RCE) do Prompt Injection sinh ra.

4. **Trình tự quyết định bắt buộc**
    Giả sử mảng `action_requests` trả về là `[tool1, tool2]`. Bắt buộc `decisions` của bạn cũng phải trả theo đúng thứ tự là `[decision_tool1, decision_tool2]`. Nếu thứ tự truyền trong mảng bị xô lệch, lệnh tiếp tục `invoke` sẽ xử lý sai kết quả của từng chức năng.
