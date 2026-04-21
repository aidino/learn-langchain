# 13. DeepAgents Sandboxes: Môi Trường Thực Thi Cách Ly

Tài liệu này dịch và giải thích các khái niệm về Sandboxes trong DeepAgents, kết hợp cùng các Best Practices áp dụng cho phiên bản `deepagents==0.5.3` (dữ liệu từ Context7).

## 1. Giới thiệu (Overview)
**Sandbox** cho phép AI agent thực thi mã nguồn (code) trong một môi trường được cách ly hoàn toàn. Thay vì để LLM tự suy đoán kết quả, backend Sandbox cung cấp các công cụ: tiêu chuẩn filesystem (như `ls`, `read_file`, `write_file`, `grep`), và tool `execute` để phân giải linh hoạt các lệnh shell (bash, python,...). Tính năng này bảo vệ mã nguồn máy chủ thật, song vẫn đảm bảo agent có "sân chơi" thật.

**Khi nào nên áp dụng Sandbox?**
- **Coding Agents:** Cho phép agent clone repository git, chạy unit test, hoặc thậm chí tạo docker-in-docker an toàn.
- **Data Analysis Agents:** Phân tích dữ liệu với Python (Pandas, Numpy), chạy các phép tính thống kê phức tạp mà agent tự cài đặt runtime.

## 2. Các Provider Sandbox Được Hỗ Trợ
LangChain và DeepAgents phát triển module này trên mô hình Plugin. Một vài Backend Sandbox nổi bật có thể cài đặt dễ dàng:
- **LangSmith Sandbox (Mặc định)**: `pip install "langsmith[sandbox]"`
- **Modal**: `pip install langchain-modal`
- **Runloop**: `pip install langchain-runloop`
- **Daytona**: `pip install langchain-daytona`
- **AgentCore**: `pip install langchain-agentcore-codeinterpreter`

### Khởi tạo cơ bản với LangSmith Sandbox
```python
from deepagents import create_deep_agent
from deepagents.backends import LangSmithSandbox
from langchain_anthropic import ChatAnthropic
from langsmith.sandbox import SandboxClient

client = SandboxClient()
ls_sandbox = client.create_sandbox(template_name="my-template")
backend = LangSmithSandbox(sandbox=ls_sandbox)

agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-6"),
    system_prompt="You are a Python coding assistant with sandbox access.",
    backend=backend,
)

try:
    # Gọi Invoke, agent sẽ sử dụng Sandbox để cài đặt / chạy pytest
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Create a Python package and run pytest"}]
    })
finally:
    # Đừng quên xoá Sandbox sau khi sử dụng để tránh dư thừa tài nguyên!
    client.delete_sandbox(ls_sandbox.name)
```

## 3. Vòng đời Sandbox (Lifecycle & Scoping)
Sandbox Lifecycle được thiết kế xoay quanh hai trường hợp (Scope) thực tế:
- **Thread-scoped (mặc định):** Một Sandbox sống và gắn liền với một đoạn Conversation (thread_id). Dữ liệu sẽ biến mất khi Thread đóng lại.
- **Assistant-scoped:** Sandbox tồn tại xuyên suốt nhiều đoạn hội thoại. Lý tưởng cho việc chia sẻ Context lâu dài giữa các session truy vấn.

## 4. Các Mẫu Tích Hợp (Integration Patterns)

### Agent in Sandbox Pattern (Chạy Agent Bên Trong Sandbox)
Agent cùng chung Container với Sandbox.
- **Ưu điểm:** Độ trễ thấp, sát với cách thức phát triển Local.
- **Nhược điểm:** Mọi thay đổi logic bạn phải Rebuild image. Quan trọng hơn, Token LLM (API Keys) **BỊ LỘ** trực tiếp vào môi trường máy chủ chạy Sandbox. Đây là rủi ro bảo mật lớn.

### Sandbox as Tool Pattern (Sandbox Là Một Công Cụ) - KHUYÊN DÙNG
Agent sống ngoài Sandbox, chỉ trỏ tay vào Sandbox thông qua Tool API (TCP/Websockets).
- **Ưu điểm:** Cô lập State API Keys trên Host của bạn. Update logic Agent thoải mái không cần restart Docker/Sandbox. Nếu Sandox gặp lỗi sập, Agent vẫn sống để report.
- **Nhược điểm:** Phải đánh đổi latency mỗi lần Agent ra lệnh `execute()`.

## 5. Thao Tác Vào Ra Qua Filesystem
DeepAgents tách riêng khái niệm **Developer** với **Agent AI** bằng hai bình diện khác nhau:
1. **Agent Tools:** Agent sẽ dùng luồng `read_file`, `write_file`, `ls`, `grep` do LLM điều khiển.
2. **Developer Operations:** Cung cấp API `upload_files()` để mồi (seeding) dữ liệu trước, hoặc `download_files()` để trích xuất báo cáo về lại máy chủ bạn.

**Ví dụ upload và download:**
```python
from daytona import Daytona
from langchain_daytona import DaytonaSandbox

sandbox = Daytona().create()
backend = DaytonaSandbox(sandbox=sandbox)

# 1. Developer load data mẫu trước cho Agent
backend.upload_files([
    ("/src/index.py", b"print('Hello Sandbox')\n"),
    ("/pyproject.toml", b"[project]\nname = 'my-sandbox-app'\n"),
])

# 2. Agent thực hiện tác vụ (Đã khởi chạy ở trên) ...

# 3. Developer Download kết quả sau hoàn thành:
results = backend.download_files(["/src/index.py", "/output.txt"])
for result in results:
    if result.content is not None:
        print(f"{result.path}: {result.content.decode()}")
```

---

## 🌟 Best Practices & Codes Cụ Thể (deepagents==0.5.3)
Theo cơ sở dữ liệu `Context7` cập nhật, sau đây là những kĩ thuật và best-practice thiết yếu cho các cấu trúc `deepagents==0.5.3`:

### 5.1 Cấu Hình Sandbox Với Định Dạ TOML
DeepAgents bản `0.5.3` ra mắt công cụ CLI `deepagents deploy`, giúp định hình Sandbox rất thanh lịch bằng File Cấu Hình (Configurations file) thay vì Hardcode.

_Ví dụ File `deepagents.toml`_
```toml
[agent]
name = "deepagents-deploy-coding-agent"
model = "anthropic:claude-sonnet-4-5"

[sandbox]
provider = "langsmith"
template = "coding-agent"
image = "python:3.12"
scope = "thread"   # Có thể thay đổi runtime
```
Khi chạy CLI: `deepagents deploy`, hệ thống tự thiết lập provider và images mà không tốn công sức.

### 5.2 Khởi Tạo Sandbox Với Runloop SDK
Cú pháp với provider `Runloop` đã được Optimize mạnh trong phiên bản này:

```python
import os
from runloop_api_client import RunloopSDK
from langchain_runloop import RunloopSandbox

# Trích xuất KEY và khai báo Runloop API client ở phiên bản mới.
client = RunloopSDK(bearer_token=os.environ["RUNLOOP_API_KEY"])
devbox = client.devbox.create()
sandbox = RunloopSandbox(devbox=devbox)

try:
    result = sandbox.execute("echo 'Running commands remotely in Runloop !'")
    print(result.output)
finally:
    devbox.shutdown()
```

### 5.3 Chuyển Đổi Động Sandbox CPU / GPU
Best practice nâng cao trong việc tối ưu chi phí hạ tầng (nhất là trong môi trường NVIDIA Deep Agent mẫu): Cho phép chuyển tiếp **động** type của Sandbox tại lệnh `.invoke`.

```python
# Yêu cầu Sandbox chạy bằng GPU Node cho task huấn luyện
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Train the PyTorch script within Sandbox"}]},
    context={"sandbox_type": "gpu"} # <-- Override configuration dynamically!
)
```

### 5.4 Quan Tâm Security Exfiltration (Chuẩn Bảo Mật)
- **Quản lý Secrets**: KHÔNG ĐƯỢC cung cấp secret token vào trong `execute`. Nếu tác vụ cần xác thực (Github, DB), hãy đặt tool kết nối đứng ở môi trường Host của bạn bên ngoài Sandbox (hoặc sử dụng Network proxy Header Injection có hỗ trợ).
- **Sức Khỏe Network Của Sandbox**: Các LLMs dễ dính ngộ nhận Context-Injection. Ở Sandbox như `Modal`, bạn nên giới hạn Network bằng `blockNetwork: true` để tránh LLM tải lên hoặc xuất File lộ bảo mật về Host lạ.
- **Tiền Kiểm & Human-In-The-Loop (HITL)**: Đặt HITL phê duyệt ở cấp `execute()`, kết hợp cùng Layer Backends chặn (Filter Middleware) để kiểm soát gắt gao quá trình này nếu hệ thống ra biển lớn (Production).

---

## 6. Mở Rộng: Tích Hợp AIO Sandbox (agent-infra/sandbox)
Bên cạnh các provider chính thức, cộng đồng OpenSource cung cấp thêm **AIO (All-in-One) Sandbox** (`agent-infra/sandbox`) - một giải pháp tự host (self-hosted) cực kì nhẹ. Điểm khác biệt lớn nhất là AIO đóng gói toàn bộ: **Browser (VNC) + File System + Terminal + Jupyter Code Server + MCP Hub** vào duy nhất 1 Docker container.

### 6.1 Tổng quan về AIO Sandbox
- **Đa phương thức (Multi-Interface):** Không chỉ cung cấp API thực thi lệnh, hệ thống cho phép giám sát trực quan (Preview Proxy) thông qua UI như `/vnc/index.html`, `/code-server/`, và `/v1/shell/ws`.
- **Hệ thống file đồng nhất (Unified FS):** File tải về từ trình duyệt nội bộ có thể lập tức được parse qua Shell/Jupyter mà không mất công sao chép (copy) tài nguyên qua lại.
- **Tích hợp MCP Server (Model Context Protocol):** Cung cấp sẵn các Hub Server cho MCP protocol. Điều này hỗ trợ Agent tự động điều khiển Trình duyệt qua CDP, thao tác Desktop hay đọc Document rất ổn định.
- **Khởi tạo siêu tốc** thông qua 1 câu lệnh Docker:
  ```bash
  docker run --security-opt seccomp=unconfined --rm -it -p 8080:8080 ghcr.io/agent-infra/sandbox:latest
  ```

### 6.2 Kết hợp AIO Sandbox với DeepAgents 
Hiện tại DeepAgents **chưa hỗ trợ sẵn** provider này như Native (giống Modal, Daytona). Tuy nhiên, theo kiến trúc *Sandbox as Tool Pattern* của DeepAgents, chúng ta sẽ viết một Custom Backend Wrapper kế thừa từ class abstract `BaseSandbox`, sau đó bọc API Client của `agent_sandbox` (từ AIO) vào bên trong.

**Ví dụ Code (Tự Build Custom Backend):**

```python
from deepagents.backends.sandbox import BaseSandbox, SandboxResult
from agent_sandbox import Sandbox as AIOSandboxSDK

class AIOSandboxBackend(BaseSandbox):
    def __init__(self, endpoint="http://localhost:8080"):
        """ Khởi tạo SDK Client của agent-infra """
        self.client = AIOSandboxSDK(base_url=endpoint)
        self.sandbox_context = self.client.sandbox.get_context()
        self.home_dir = self.sandbox_context.home_dir
        
    def execute(self, command: str, **kwargs) -> SandboxResult:
        """ DeepAgents Protocol yêu cầu hàm này cho việc thực thi lệnh shell """
        try:
            # Gửi lệnh thực thi tới AIO Docker Container
            res = self.client.shell.exec_command(command=command)
            # Trả về kết quả đầu ra theo chuẩn của Pipeline DeepAgents 
            return SandboxResult(output=res.data.output, error=None)
        except Exception as e:
            return SandboxResult(output="", error=str(e))

    def read_file(self, filepath: str) -> str:
        """ Cung cấp cho LLM file thông qua Sandbox API """
        res = self.client.file.read_file(file=filepath)
        return res.data.content
        
    # (Developer có thể định nghĩa tương tự cho write_file, ls, v.v...)
```

**Cách gọi Agent sử dụng Sandbox tự thiết kế trên:**
```python
from deepagents import create_deep_agent

# Khởi tạo custom Sandbox kết nối với Local Docker Container ở HD 6.1
aio_sandbox = AIOSandboxBackend(endpoint="http://localhost:8080")

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    backend=aio_sandbox,
    system_prompt="Bạn là Assistant. Khi cần chạy code hoặc duyệt web, hãy chạy trong môi trường Sandbox",
)

result = agent.invoke({"messages": [{"role": "user", "content": "Liệt kê files trong ~/ bằng ls -la"}]})
```

Ngay lúc này không chỉ agent được hưởng lợi từ việc gọi code an toàn do DeepAgent lo, mà bản thân Developer cũng có thể vào `localhost:8080/index.html` của AIO để xem trực tiếp Agent đang cài đặt cái gì thông qua công nghệ VNC tích hợp!
