# Quyền Hạn (Permissions) - Deep Agents

**Mô tả**: Kiểm soát quyền truy cập hệ thống tệp (filesystem) bằng các quy tắc cấp quyền dạng khai báo (declarative) cho Deep Agents.
**Nguồn**: https://docs.langchain.com/oss/python/deepagents/permissions

Theo mặc định, Deep Agents chạy với một số công cụ hệ thống tệp như `ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`. Trên các môi trường sandbox (hộp cát), agent còn có công cụ `execute`. Hệ thống quyền (`permissions`) cho phép cấu hình kiểm soát truy cập đọc/ghi các thư mục cụ thể thông qua các quy tắc linh hoạt, kết hợp cùng các policy hook của backend hệ thống.

## Cách sử dụng cơ bản

Dùng lớp `FilesystemPermission` để xác định danh sách các quy tắc, sau đó truyền vào hàm `create_deep_agent`:

```python
from deepagents import create_deep_agent, FilesystemPermission

# Agent chỉ đọc: từ chối mọi thao tác ghi trên toàn bộ hệ thống
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

## Cấu trúc quy tắc (Rule structure)

Một quy tắc `FilesystemPermission` gồm các thành phần sau:
- `operations`: Nhận một danh sách chuỗi thao tác `["read"]` hoặc `["write"]`
  - Hành động `"read"` (đọc) bao gồm các công cụ: `ls`, `read_file`, `glob`, `grep`.
  - Hành động `"write"` (ghi) bao gồm các công cụ: `write_file`, `edit_file`.
- `paths`: Danh sách các chuỗi đường dẫn (list[str]), ví dụ: `["/workspace/**"]`. Hệ thống kiểm soát này hỗ trợ dấu sao `**` cho các thư mục con và cú pháp gom cụm với dấu ngoặc `{a,b}`.
- `mode`: Bật chế độ `"allow"` (Cho phép) hay `"deny"` (Từ chối). Quy định có hiệu lực khi cả loại thao tác và đường dẫn đều đã khớp.

## Các ví dụ chuẩn

### 1. Cách ly vào một thư mục không gian làm việc (Isolate to a workspace directory)
Dành riêng quyền thao tác chỉ được thực hiện trong `/workspace/`:
```python
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/workspace/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

### 2. Bảo vệ các tệp tin cụ thể (Protect specific files)
Nguyên tắc từ chối đọc/ghi đối với danh sách tệp định dạng ẩn hoặc quan trọng trong workspace:
```python
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/workspace/.env", "/workspace/examples/**"],
            mode="deny",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/workspace/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

### 3. Bộ nhớ chỉ đọc (Read-only memory)
Thường được áp dụng khi agent cần tham chiếu thông tin cấu hình đọc từ bộ nhớ chính (memories) hay các file cấu hình chính sách (policies) nhưng lại không được quyền thay đổi chúng:
```python
agent = create_deep_agent(
    model=model,
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
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/memories/**", "/policies/**"],
            mode="deny",
        ),
    ],
)
```

### 4. Từ chối tất cả quyền truy cập (Deny all access)
```python
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

### 5. Thứ tự cấp quyền (Rule ordering)
Hệ thống cấp quyền đánh giá quy định theo thứ tự định nghĩa (từ trên xuống dưới đầu mảng). Quy tắc khớp đầu tiên luôn làm quy tắc hoạt động. Hãy luôn đặt quy định ưu tiên cao (mật độ hẹp) ở đầu phần tử.
```python
# Sai lầm: /workspace/** quét dính luôn /workspace/.env đầu tiên, do đó quy định từ chối sửa .env không bao giờ được chạm tới.
permissions=[
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/**"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/.env"],
        mode="deny", # Không bao giờ thực thi (Never reached)
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**"],
        mode="deny",
    ),
]
```

### 6. Quyền của Agent phụ (Subagent permissions)
Các agent phụ (Subagents) được tạo ra không tự động chép quyền kế thừa từ agent điều phối chính. Bạn cần khai báo rõ hệ thống quyền khai báo cho riêng subagent đối nội với thông số `permissions`:
```python
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/workspace/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
    subagents=[
        {
            "name": "auditor",
            "description": "Read-only code reviewer",
            "system_prompt": "Review the code for issues.",
            "permissions": [
                FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
                FilesystemPermission(operations=["read"], paths=["/workspace/**"], mode="allow"),
                FilesystemPermission(operations=["read"], paths=["/**"], mode="deny"),
            ],
        }
    ],
)
```

### 7. Backend tổng hợp (Composite backends)
Khi sử dụng công nghệ `CompositeBackend`, quyền hạn sẽ được quản lý dựa theo từng tuyến truy cập backend độc lập (scoped to route). Việc cấp quyền chồng chéo không chỉ rõ scope trong cơ sở này sẽ gây ra lỗi `NotImplementedError`. Hệ thống quyền sẽ chặn định tuyến bao trùm của toàn cục mặc định.

---

## 🚀 Thực hành Tốt nhất (Best Practices) với `deepagents==0.5.3` (Trích xuất bổ sung từ Context7)

Theo hướng dẫn từ thư viện `deepagents==0.5.3`, khi xử lý tính năng **Bảo mật và Phân quyền (Permissions)**, nguyên tắc thực hành tốt nhất (Best Practice) là thiết kế nó giao thoa chặt chẽ với cơ chế **Human-in-the-Loop (HITL - Chờ phê duyệt của con người)**. Việc phân quyền dạng khai báo ở lõi là chưa đủ trên môi trường production nếu các công cụ thay đổi dữ liệu bị thiếu khâu kiểm duyệt.

Dưới đây là một pattern kết hợp để tạo ra hệ thống an toàn nhất: 

### Kết hợp cấu trúc Phân Quyền tĩnh với HITL động qua `interrupt_on`
Trong bản `0.5.3`, khi cấp quyền truy xuất file cứng, bạn hãy dùng thêm cơ chế Checkpointer với điều hướng `interrupt_on`. Nó cho phép Agent tạm ngừng, tạo ra tín hiệu yêu cầu cấp quyền người dùng với các API rủi ro cao mặc dù truy cập là hơp lệ.

```python
from deepagents import create_deep_agent, FilesystemPermission
from langgraph.checkpoint.memory import MemorySaver

# Sử dụng MemorySaver để lưu trữ trạng thái khi ngắt quy trình
checkpointer = MemorySaver()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    checkpointer=checkpointer,
    
    # TẦNG 1: QUYỀN HẠN KHAI BÁO (Từ chối ngay từ Backend level)
    permissions=[
        FilesystemPermission(operations=["read", "write"], paths=["/workspace/public/**"], mode="allow"),
        FilesystemPermission(operations=["read"], paths=["/workspace/secrets/**"], mode="deny"),
        FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
    ],
    
    # TẦNG 2: HUMAN-IN-THE-LOOP (Chặn hoạt động cho phép để xin cấp quyền từ cấp Quản Trị)
    interrupt_on={
        "write_file": True,           # Đợi người quản trị duyệt trước khi ghi nội dung
        "edit_file": True,            # Mọi yêu cầu cấu trúc sửa đổi sẽ sinh trễ đợi phê duyệt
        "execute": {"always": True},  # Bắt buộc có người duyệt cho mọi dòng lệnh bash execute
    },
)

# 1. Khởi chạy và đánh dấu session
config = {"configurable": {"thread_id": "auditor-deploy-session"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Tiến hành sửa đổi file cấu hình config.yaml trong public"}]},
    config=config
)

# 2. Bắt Tín Hiệu Xin Cấp Quyền (HITL Interruption Hook)
if result.get("__interrupt__"):
    print("Agent tạm dựng yêu cầu Cấp Chặn Cấp Quyền Thực Thi/Sửa Đổi:", result["__interrupt__"])
    
    # Để xác nhân phê duyệt theo form (Approve Permissions):
    # result = agent.invoke(None, config=config)
```

**Tại sao đây là Best Practice trong phiên bản `0.5.3`?**
- `Permissions` tĩnh chỉ có mức độ bảo vệ tệp tin, nó không kiểm soát được tác động của nội dung bên trong file hay script do logic LLM sinh ra.
- Kết hợp cả hai cung cấp sự đảm bảo 2 lớp (Defense In Depth): Backend Permissions từ chối thẳng thừng các luồng truy cập không hợp lệ;  Và `interrupt_on` đảm bảo những truy cập hợp lệ vào luồng xử lý chính vẫn không thể tự đồng sửa đổi dữ liệu nếu chưa có thao tác phê chuẩn trên giao diện Front-end.
- Ngoài ra, bạn cần truyền `checkpointer` (ví dụ `MemorySaver()`) thì trạng thái biểu đồ HITL mới được hệ thống ghi nhận.
