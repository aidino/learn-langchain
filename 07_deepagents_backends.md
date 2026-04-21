# Quản lý Backends trong DeepAgents

Tài liệu này giải thích cách chọn, cấu hình và sử dụng filesystem backends (hệ thống lưu trữ) cho DeepAgents. Backends cho phép agent tương tác với dữ liệu, đọc/ghi file, và quản lý persistent storage (lưu trữ bền vững) theo các chính sách bảo mật chặt chẽ.

---

## 1. Tổng quan các loại Backend tích hợp sẵn (Built-in backends)

DeepAgents cung cấp sẵn nhiều loại "môi trường lưu trữ" phù hợp cho từng giai đoạn từ phát triển (development) đến sản xuất (production).

### 1.1 StateBackend (Lưu trữ tạm thời / Ephemeral)
Đây là backend **mặc định** khi bạn khởi tạo agent.
- **Cơ chế:** Lưu trữ file trực tiếp bên trong Agent State của LangGraph cho thread hiện tại.
- **Đặc điểm:** Tồn tại xuyên suốt các lượt hội thoại (turns) trong cùng một thread thông qua checkpoint. Hoạt động như một "tờ giấy nháp" (scratch pad) cho agent lưu trữ kết quả trung gian.
- **Sử dụng:** `backend=StateBackend()`

### 1.2 FilesystemBackend (Lưu trữ trên đĩa thật cấu hình Local)
Cho phép agent đọc/ghi trực tiếp lên ổ cứng thật của hệ thống.
- **Cơ chế:** Đọc và ghi các file thật dưới một thư mục gốc (`root_dir`) định trước.
- **Bảo mật quan trọng:** Bằng cách thiết lập `virtual_mode=True`, agent sẽ bị giới hạn chỉ hoạt động ở trong thư mục `root_dir` (ngăn chặn các path traversal nhạy cảm như `..`, `~` hay absolute paths bên ngoài).
- **Trường hợp sử dụng an toàn:** Môi trường Local hoặc các container / sandbox cho CI/CD. Đừng bỏ `FilesystemBackend` chứa biến số môi trường `.env` hay config nhạy cảm mà không bật `virtual_mode`.

```python
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    backend=FilesystemBackend(root_dir="./", virtual_mode=True)
)
```

### 1.3 LocalShellBackend (Tương tác Local Shell)
Mở rộng từ `FilesystemBackend`, cung cấp thêm công cụ `execute` để agent chạy các lệnh shell trực tiếp trên máy chủ (`subprocess.run(shell=True)`).
- **Cảnh báo Bảo mật (Threat Model):** Lệnh được chạy không qua sandbox, giới hạn trực tiếp tuỳ thuộc vào host user. TUYỆT ĐỐI không dùng ở Production.
- **Best Practice:** Chỉ dùng cho local dev tools được tin tưởng HOẶC phải tích hợp `Human-in-the-loop` (HITL middleware) để con người duyệt từng command một trước khi chạy. Sandbox thực thụ (Platform-specific sandboxes) nên được ưu tiên hơn.

### 1.4 StoreBackend (Lưu trữ bền vững với LangGraph Store)
Lưu trữ file dài hạn (cross-thread) bằng LangGraph `BaseStore` (hỗ trợ nền tảng InMemory, Redis, Postgres hoặc LangSmith Deployment).
- **Cơ chế phân tách (Namespace factories):** StoreBackend phân tách file dựa trên namespace. 
- **Lưu ý cập nhật ở DeepAgents >= 0.5.3:** API đã thay đổi để dùng tham số `Runtime` (rt) thay cho `BackendContext` (đã deprecated).
  
```python
from deepagents.backends import StoreBackend

# Tạo storage riêng biệt cho từng người dùng:
backend = StoreBackend(
    namespace=lambda rt: (rt.server_info.user.identity,)
)
```

### 1.5 CompositeBackend (Bộ định tuyến Router)
Cho phép kết hợp đa backend thành một cấu trúc thư mục ảo duy nhất. Filesystem tool sẽ định tuyến (route) prefix thư mục với backend xử lý tương ứng (tài liệu tìm kiếm cho thấy tiền tố dài hơn sẽ ghi đè).

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

backend = CompositeBackend(
    default=StateBackend(), # Lưu nháp vào LangGraph State
    routes={
        "/memories/": StoreBackend(namespace=lambda rt: (rt.server_info.user.identity,)), # Bộ nhớ dài hạn trên Store cross-thread
        "/projects/": FilesystemBackend(root_dir="/workspace/projects", virtual_mode=True)
    }
)
```

---

## 2. Bảo mật và Quản lý Quyền Truy cập (Permissions & Policy Hooks)

### 2.1 Sử dụng FilesystemPermission
Bạn có thể giới hạn quyền của Agent bằng cách chặn hoặc chỉ cho phép (allow/deny mode) ở những prefix nhất định thông qua `create_deep_agent(..., permissions=[...])`.

```python
from deepagents import FilesystemPermission

permissions = [
    FilesystemPermission(
        operations=["write", "edit"], 
        paths=["/policies/**"], 
        mode="deny"
    )
]
```

### 2.2 Tạo Policy Wrapper (Hook chặn tùy biến)
Wrap layer của `BackendProtocol` để chặn truy cập ở mức backend:
```python
class PolicyWrapper(BackendProtocol):
    def __init__(self, inner: BackendProtocol, deny_prefixes: list[str]):
        self.inner = inner
        self.deny_prefixes = [p if p.endswith("/") else p + "/" for p in deny_prefixes]
        
    def write(self, file_path: str, content: str) -> WriteResult:
        if any(file_path.startswith(p) for p in self.deny_prefixes):
            return WriteResult(error=f"Việc ghi bị chặn ở {file_path}")
        return self.inner.write(file_path, content)
    # Tương tự với edit(), ls(), read(), ...
```

---

## 3. Tạo Hệ Thống Virtual Filesystem Riêng
Bạn có thể thiết lập Cloud Storage riêng như S3 hoặc Postgres bằng cách kế thừa `BackendProtocol`:
- Bắt buộc tuân theo giao thức: Trả về Object cụ thể (`LsResult`, `WriteResult`, `ReadResult`) có trường `error` nếu lỗi thay vì Raise Exception.
- Có sắn các hàm: `ls(path)`, `read(file_path)`, `grep(pattern)`, `glob(pattern)`, `write()`, và `edit()`.

---

## 4. Best Practices & Tips với DeepAgents 0.5.3+

Dựa trên Context7, đây là những pattern quan trọng khi ứng dụng bản mới:

1. **Hiểu rõ Threat Model Defaults (Mô hình rủi ro):**
   Mặc định, `create_deep_agent` chạy **`StateBackend`** rất an toàn vì agent chỉ lưu trữ file trong bộ nhớ tạm của thread đó, không bị tràn memory và không thể thao túng OS. 

2. **Cách tiêm Memory (Tích hợp State/Store Backend với Middleware Hệ Thống):**
   Trong kiến trúc middleware, Memory và Skills có thể được load từ FilesystemBackend vào System Prompt.
   ```python
   agent = create_deep_agent(
       memory=["/AGENTS.md"], # Lấy từ root_dir load thẳng vào System Prompt
       skills=["/skills/"],
       backend=FilesystemBackend(root_dir="./", virtual_mode=True),
       subagents=load_subagents("./subagents.yaml") # Subagent truyền ngoài middleware
   )
   ```

3. **Gỡ bỏ code cũ (`BackendContext`):**
   Từ version `0.5.2` đổ đi, cách truyền factory `lambda ctx: StoreBackend(...)` đã lỗi thời và sử dụng `BackendContext`. Khuyên dùng `Runtime` như mẫu dưới:
   - *Thay vì:* `ctx.runtime.context.user_id`
   - *Dùng:* `rt.server_info.user.identity` hoặc `rt.context.org_id`

4. **Tích hợp linh hoạt với `FilesystemMiddleware` (khi xài Raw LangGraph agent):**
   Nếu bạn đang định nghĩa LangGraph Agent bằng Langchain cơ sở (`create_agent`), bạn có thể gói riêng `FilesystemMiddleware` thay vì wrapper của DeepAgents:
   ```python
   from deepagents.middleware.filesystem import FilesystemMiddleware
   
   agent = create_agent(
       model="anthropic:claude-sonnet-4-6",
       middleware=[FilesystemMiddleware(backend=StateBackend())],
   )
   ```
