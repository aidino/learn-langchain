# Triển khai (Deploy) Agent với Deep Agents CLI

Tài liệu này hướng dẫn cách đưa một AI agent (hoạt động độc lập với model cụ thể, mã nguồn mở) lên môi trường production sử dụng công cụ dòng lệnh (CLI) của Deep Agents.

## 1. So sánh với Claude Managed Agents (Anthropic)
Trong khi Claude Managed Agents bị trói buộc với các model của Anthropic và môi trường do Antropic quản lý, thì Deep Agents cho phép:
- Sử dụng **bất kỳ model nào** từ các nhà cung cấp khác nhau (OpenAI, Anthropic, Google, v.v.).
- Sử dụng **bất kỳ môi trường sandbox nào** (như Daytona, Modal, Runloop, LangSmith hoặc tự build).

## 2. Kiến trúc của một Deep Agent (Những gì bạn sẽ deploy)
Khi chạy lệnh `deepagents deploy`, bạn đang đóng gói và triển khai một dự án chuẩn bao gồm:
- **Model**: Được chỉ định trong file cấu hình.
- **`AGENTS.md`**: File chứa "System prompt" - hướng dẫn chính cho Agent gốc.
- **`skills/`**: Thư mục chứa các kỹ năng (công cụ/hành động) được viết theo chuẩn Agent Skills.
- **`user/`**: Thư mục chứa bộ nhớ cá nhân hóa cho từng người dùng, agent có quyền ghi vào đây để học thói quen người dùng qua thời gian.
- **`mcp.json`**: File cấu hình chứa các kết nối tới các MCP server (Model Context Protocol) cho phép agent sử dụng công cụ từ bên ngoài.
- **`subagents/`**: Thư mục chứa định nghĩa các Agent con để agent chính có thể phân bổ công việc.
- **`deepagents.toml`**: File cấu hình của dự án.
- **Sandbox**: Môi trường an toàn độc lập mà agent có thể thực thi mã code và thao tác file.

## 3. Cài đặt và Sử dụng CLI
**Cài đặt từ dòng lệnh bằng uv:**
```bash
uv tool install deepagents-cli
```

**Các lệnh cơ bản:**
- `deepagents init [name]`: Khởi tạo một dự án agent khởi điểm mới với tên tham số được truyền vào.
- `deepagents dev`: Chạy và thử nghiệm agent trên máy local (mặc định ở port 2024).
- `deepagents deploy`: Đóng gói và lên lịch triển khai agent lên LangSmith Deployment (production). Có thể cấu hình thêm file thông qua cờ `--config path/to/deepagents.toml`.

## 4. Cấu trúc Dự Án Mẫu
```text
my-agent/
├── deepagents.toml       # Cấu hình dự án
├── AGENTS.md             # File Prompt chính của tác nhân
├── .env                  # Biến môi trường
├── mcp.json              # Kết nối với MCP
├── skills/               # Thư mục công cụ
│   ├── code-review/
│   │   └── SKILL.md
│   └── data-analysis/
│       └── SKILL.md
├── subagents/            # Quản lý sự phối hợp Subagents
│   └── researcher/
│       ├── deepagents.toml
│       └── AGENTS.md
└── user/                 # Bộ Nhớ cá nhân
    └── AGENTS.md
```

## 5. File cấu hình `deepagents.toml`
Đây là trung tâm cấu hình của Agent.
### Phần `[agent]`
Định nghĩa thông tin định danh và loại model của tác nhân.
```toml
[agent]
name = "research-assistant"
model = "google_genai:gemini-3.1-pro-preview" # Cú pháp <provider>:<model>
```
*Lý thuyết đóng gói:* Deep Agents CLI sẽ tự động dịch provider thành package library liên kết tương ứng (ví dụ: `google_genai` cần cài `langchain-google-genai`).

### Phần `[sandbox]`
Quy định môi trường thực thi code của tác nhân.
```toml
[sandbox]
provider = "daytona" # Hoặc "modal", "langsmith", "runloop", "none"
template = "coding-agent"
image = "python:3.12"
scope = "assistant" # Hoặc "thread"
```
*Giải thích về `scope`:*
- `"thread"` (Mặc định): Mỗi phiên chat (conversation) sẽ khởi động một sandbox sạch sẽ. Khác phiên thì sẽ sử dụng file system cách ly vật lý.
- `"assistant"`: Dùng chung một cấu trúc dữ liệu sandbox cho mọi cuộc hội thoại. State và file tải xuống được giữ nguyên vĩnh viễn.

## 6. File `.env` (Biến môi trường)
Chứa API key cần thiết để tương tác với LLM và triển khai:
```env
# Key cấu hình nhà cung cấp LLMs
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Key Deploy lên LangSmith
LANGSMITH_API_KEY=lsv2_...

# API Token cho các Sandbox Provider
DAYTONA_API_KEY=...
MODAL_TOKEN_ID=...
```

## 7. Hỗ trợ Integration Khác (Deployment endpoints)
Khi đã đưa lên production, Deep Agents của bạn có khả năng hỗ trợ gọi kết nối vào agent của bạn thông qua:
- **MCP (Model Context Protocol)**: Biến Agent của bạn thành Tool cho phép một Agent bên thứ ba gọi xử lý.
- **A2A (Agent to Agent)**: Điều hướng nhiều Agent đa luồng giao tiếp.
- **Agent Protocol**: API theo chuẩn công nghiệp để liên kết trực tiếp ứng dụng Frontend UI vào bot.

## 8. Đặc tính Kỹ thuật Nổi bật
### A. User Memory (Bộ nhớ người dùng)
Agent có khả năng ghi nhớ thông tin riêng biệt cho từng người dùng cá nhân để cung cấp trợ lý "đo ni đóng giày":
- File `user/AGENTS.md` hoạt động như một System Prompt cá nhân. Nếu folder có tồn tại, một bản copy sẽ được tạo riêng trong namespace `/memories/user/AGENTS.md` tương ứng từng User ID khởi tạo. 
- Agent được cấp công cụ `edit_file` thông minh giúp nó chủ động lưu lại thói quen, công việc chưa hoàn thành lên memory space này. Ngược lại, dữ liệu hệ thống như `skills/` được giới hạn bảo mật **chỉ đọc (read-only)**.

### B. Mạng lưới Subagents (Các Agent Phụ)
Giúp chia để trị công việc phức tạp:
- Khai báo các mô-đun cấp thấp bên dưới `subagents/`.
- Khi cấu hình đúng chuẩn, CLI tự động tiêm một công cụ gọi là `task` cho Tác nhân Xuyên tâm (Main Agent). Nhiệm vụ của nó là phát yêu cầu và giao việc cho cấp dưới theo chỉ dẫn trong System prompt của Subagent đo.
- Subagent có cấu hình thư mục kỹ năng/mô hình phân tầng riêng, nhưng nó sẽ *kế thừa tự động* môi trường Sandbox lẫn API Model từ Main Agent trừ khi bị khai báo ép kiểu ghi đè. Không gian nhớ của từng Subagent được chạy trên Sandbox chéo độc lập (Memory Isolation) tách biệt luồng nhớ `/memories/subagents/<name>/**` khỏi User Space.

## 9. Những hạn chế hiện tại (Limitations)
- **Cấu hình MCP Interface**: Bản dựng thông qua Bundle CLI cho deploy chỉ hỗ trợ phương thức kết nối Tool `HTTP` và `SSE`. Nếu `mcp.json` cấu hình sử dụng phương thức thông qua Cổng chuẩn `stdio`, tiến trình sẽ văng lỗi chặn build lập tức.
- **Khả năng Lập Trình Python Tools Trực Tiếp**: Tác nhân của LangChain hiện tại không trực tiếp cho phép truyền chuỗi Python Runtime custom bằng hàm trực tiếp (như Tool Component trên LangGraph thông thường). Giải pháp bắt buộc là sử dụng triển khai qua chuẩn MCP Servers.
