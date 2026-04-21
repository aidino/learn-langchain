# DeepAgents Skills

Tài liệu này giải thích cách mở rộng khả năng của agent bằng cách sử dụng **Skills** (kỹ năng) cung cấp hướng dẫn và ngữ cảnh chuyên biệt. Nó bao gồm các best practice dành cho **deepagents==0.5.3**.

## Tổng quan (What are skills)
Skills là một cách để đóng gói các thủ tục, kiến thức dạng quy trình, hay tài liệu hướng dẫn chuyên môn để agent có thể pull (kéo) về sử dụng chỉ khi cần thiết, giúp giữ system prompt (ngữ cảnh gốc) luôn ngắn gọn thay vì nhồi nhét mọi thứ.
Một bộ skill điển hình có cấu trúc thư mục bao gồm:
- Tệp `SKILL.md` chứa hướng dẫn và metadata (bắt buộc).
- Các script bổ sung (tùy chọn).
- Thông tin tham khảo như tài liệu (tùy chọn).
- Các asset như template hay tài nguyên khác (tùy chọn).

## Tệp SKILL.md (Cấu trúc Best Practice)
Theo best practice cho DeepAgents 0.5.3, tệp định nghĩa kỹ năng (`SKILL.md`) nên có cấu trúc như sau:
1. **YAML Frontmatter:** Bao gồm `name` (tên kỹ năng) và `description` (khi nào agent nên dùng). Nếu description dài quá 1024 ký tự sẽ bị truncate. Bạn cũng nên thêm `license`, `compatibility`, `metadata` (author, version, allowed-tools) tùy nhu cầu.
2. **Overview (Tổng quan):** Giải thích ngắn gọn về kỹ năng.
3. **Best Practices (Thực hành tốt nhất):** Ưu tiên đưa các ghi chú best practice lên phần đầu, giải thích tại sao nên làm.
4. **Process / Instructions (Quy trình):** Liệt kê các bước thực hiện ở dạng mệnh lệnh (imperative form) một cách súc tích.
5. **Common Pitfalls (Các lỗi thường gặp):** Đưa ra các anti-pattern (cách làm sai) để agent tránh những sai lầm có thể xảy ra.

*(Dung lượng tệp `SKILL.md` không được vượt quá 10 MB, các tệp lớn hơn sẽ bị bỏ qua khi load skill).*

### Ví dụ Hoàn Chỉnh (Full Example)
```yaml
---
name: langgraph-docs
description: Dùng skill này cho các yêu cầu liên quan đến LangGraph để lấy tài liệu hướng dẫn chính xác.
license: MIT
compatibility: Yêu cầu truy cập internet
metadata:
  author: langchain
  version: "1.0"
  allowed-tools: fetch_url
---
# langgraph-docs

## Overview
Kỹ năng này giải thích cách truy cập tài liệu Python của LangGraph để hướng dẫn triển khai.

## Instructions
### 1. Fetch the documentation index
Dùng tool `fetch_url` để đọc: `https://docs.langchain.com/llms.txt`. Nó cung cấp danh sách tài liệu có sẵn.

### 2. Select relevant documentation
Dựa trên câu hỏi, chọn 2-4 URL phù hợp nhất từ index. Ưu tiên:
- Các hướng dẫn "how-to" cho việc triển khai.
- Các "core concept" để hiểu nguyên lý.
- Tutorials cho ví dụ hoàn chỉnh (end-to-end).
- Reference cho chi tiết API.

### 3. Fetch selected documentation
Dùng `fetch_url` đọc các mục tài liệu đã chọn.

### 4. Provide accurate guidance
Hoàn thành yêu cầu của user dựa trên tài liệu đã đọc.
```

## Cách Agent sử dụng Skills (What the agent sees)
Quá trình agent nhận diện và sử dụng skill diễn ra theo 3 bước:
1. **Match (So khớp):** Khi có prompt từ user, agent kiểm tra xem `description` của skill nào khớp và phù hợp với tác vụ.
2. **Read (Đọc):** Nếu skill ấy phù hợp, agent sẽ đọc toàn bộ nội dung tệp `SKILL.md` theo đường dẫn skill tương ứng đã load.
3. **Execute (Thực thi):** Agent làm theo từng bước hướng dẫn trong skill và sử dụng các file đính kèm (script, template, tài liệu) khi cần thiết.

## Sử dụng trong Code (Usage in deepagents 0.5.3)

Bạn có thể truyền resources qua nhiều driver backend: `StateBackend` (mặc định), `StoreBackend`, hoặc `FilesystemBackend`.

### Sử dụng với FilesystemBackend (Thực tế & Khuyên dùng)
Skills được load trực tiếp từ ổ đĩa theo đường dẫn tương đối so với thư mục `root_dir` của backend.

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend

# Checkpointer BẮT BUỘC NẾU có human-in-the-loop (như cơ chế interrupt)
checkpointer = MemorySaver()

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    backend=FilesystemBackend(root_dir="/Users/user/project"),
    # Cung cấp URI/thư mục tới nơi chứa skills
    skills=["/Users/user/project/skills/"],
    interrupt_on={
        "write_file": True,
        "read_file": False,
        "edit_file": True
    },
    checkpointer=checkpointer,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is langgraph?"}]},
    config={"configurable": {"thread_id": "12345"}}
)
```

## Source Precedence (Mức độ ưu tiên của các nguồn)
Nếu có nhiều khối cấu hình skills trong list đường dẫn và có các phân lớp skill trùng tên, **bản sao skill từ đường dẫn nằm cuối danh sách (load sau cùng) sẽ chiến thắng và ghi đè các skill trước đó.**

```python
# Nếu cả hai thư mục đều có kỹ năng tên "web-search",
# kỹ năng tải lên từ "/skills/project/" sẽ được ưu tiên.
agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    skills=["/skills/user/", "/skills/project/"],
)
```

## Skills cho Subagents
Đứng ở góc độ triển khai quản lý tác vụ phân tầng:
- **General-purpose subagent (Sub-agent chung):** Tự động kế thừa tất cả skills từ *main agent* (agent cha) khi thiết lập bằng `create_deep_agent`. Không cần cấu hình thêm.
- **Custom subagents (Sub-agent tuỳ chỉnh riêng):** *Không tự động kế thừa* skills của agent cha. Bạn phải khai báo cụ thể những skills riêng nào dùng chung qua thuộc tính `skills` trong định nghĩa subagent để ép định hướng khả năng hoạt động.

```python
research_subagent = {
    "name": "researcher",
    "description": "Subagent phân tích nghiên cứu",
    "system_prompt": "You are a researcher.",
    "tools": [web_search],
    "skills": ["/skills/research/", "/skills/web-search/"], # Nó chỉ nạp chính xác các skill này
}
```

## Skills vs Memory vs Tools
- **Skills vs Memory:**
  - `memory` (ví dụ file `AGENTS.md`) dùng để lưu trữ thông tin về bối cảnh lâu dài, sự kiện, yêu cầu cá nhân hóa (ví dụ: "Người dùng thích code luôn phải sinh block comment ngắn gọn"). Nhóm tin này quan trọng nên Agent ưu tiên lưu và mang theo mọi lúc.
  - `skills` (`SKILL.md`) dùng cho **procedural knowledge** (kiến thức dạng quy trình, hay thủ tục). Agent chỉ ném vào bối cảnh nội dung của skill đó khi nó cảm thấy cần thực hiện một thủ tục phức tạp mà `description` nhắc đến.
- **Tools vs Skills:**
  - Hãy **dùng Skills** thay vì Tools khi: Thủ tục có quá nhiều chi tiết hướng dẫn để cấu trúc lại thành tool description, agent được phép thao tác file system để truy cập template hướng dẫn đính kèm, hoặc bạn muốn gộp gom một chu trình nghiệp vụ lại với nhau.
  - Hãy **dùng Tools** khi quy trình kết nối với môi trường ngoại vi bên ngoài FileSystem mà Agent không có khả năng với tới, ví dụ gõ lệnh hệ thống hay kết nối database ngoài.

## Best Practice Tạo Kỹ Năng từ CLI
Với phiên bản `deepagents==0.5.3`, người dùng được khuyến khích tạo template kỹ năng khởi tạo từ DeepAgents Command Line Interface (CLI):
- Tạo skill ở cấp độ dự án trong `project skills directory` dành riêng cho repo:
  ```bash
  deepagents skills create <skill-name> --project
  ```
- Tạo skill dạng user local namespace để chia sẻ giữa các dự án:
  ```bash
  deepagents skills create <skill-name>
  ```
Thay vì phải nhớ lại đầy đủ cấu trúc YAML Frontmatter, lệnh này sẽ bootstrap đầy đủ format, đảm bảo quá trình thiết kế agent luôn chuẩn mực.
