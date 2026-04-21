# DeepAgents - Các Khả Năng Cốt Lõi (Harness Capabilities)

Tài liệu này dịch và giải thích các khả năng cốt lõi (harness capabilities) của framework DeepAgents, dựa trên tài liệu gốc từ LangChain.

Nguồn: [Harness capabilities - Docs by LangChain](https://docs.langchain.com/oss/python/deepagents/harness)

---

## 1. Khả năng Lập Kế Hoạch (Planning capabilities)
Công cụ chính: `write_todos`

- **Theo dõi đa tác vụ:** Có thể theo dõi trạng thái của nhiều tác vụ khác nhau (như `'pending'`, `'in_progress'`, `'completed'`).
- **Lưu trữ trạng thái:** Các danh sách việc cần làm (todos) được lưu trữ vào trong state của agent.
- **Tổ chức công việc phức tạp:** Giúp agent tổ chức và quản lý các công việc mang tính đa bước phức tạp.
- Rất hữu dụng cho các tác vụ chạy theo thời gian dài và cần kế hoạch chi tiết.

## 2. Truy Cập Hệ Thống Tập Tin Ảo (Virtual filesystem access)
Các công cụ được cung cấp mặc định:
- `ls`, `read_file`, `write_file`, `edit_file`, `glob` (ví dụ: `**/*.py`), `grep`.
- Khả năng làm việc đa phương thức (Multimodal): Hỗ trợ đọc các tệp hình ảnh (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.heic`, `.heif`), video (`.mp4`, `.mov`, `.avi`, v.v.), âm thanh (`.wav`, `.mp3`, v.v.), và tài liệu hiển thị (`.pdf`, `.ppt`, `.pptx`).

## 3. Phân Quyền Hệ Thống Tập Tin (Filesystem permissions)
Bảo mật hệ thống tập tin nội bộ khi Agent truy cập:
- Truỳen danh sách các quy tắc (rules) vào tham số `permissions=` khi khởi tạo agent.
- Mỗi luật quy định:
  - **Operations** (Hành động): "read" hoặc "write".
  - **Paths** (Đường dẫn): Đường dẫn dạng glob.
  - **Mode** (Chế độ): "allow" (cho phép) hoặc "deny" (từ chối).
- **Quy tắc ưu tiên:** Luật (rule) nào khớp đầu tiên sẽ được áp dụng. Nếu không có luật nào khớp, tác vụ được cho phép một cách mặc định.
- **Công dụng:**
  - Giới hạn agent chỉ được hoạt động trong một thư mục nhất định (ví dụ: `/workspace/`).
  - Bảo vệ các tệp tin nhạy cảm (ví dụ: `.env`, token/credentials).
  - Cung cấp cho các subagent (đại lý phụ) quyền hạn hẹp hơn so với agent cha.

## 4. Ủy Quyền Tác Vụ (Task delegation - Subagents)
Công cụ chính của agent: `task`
- **Cách hoạt động:** Khi agent chính gọi công cụ `task`, nó sẽ tạo ra một bản thể (instance) agent hoàn toàn mới với context độc lập. Subagent này sẽ thực thi một cách tự trị cho đến khi xong việc rồi trả về **duy nhất một bản báo cáo cuối cùng** cho agent cha.
- **Cách ly Ngữ Cảnh (Context isolation):** Công việc của subagent không làm lộn xộn context window của agent chính.
- **Thực thi song song (Parallel execution):** Nhiều subagent có thể được chạy đồng thời.
- **Đặc tả (Specialization):** Subagents có thể có các tools / cấu hình riêng biệt với agent chính (Ví dụ: `code-reviewer`, `web-researcher`, `test-runner`). Tham số để tùy chỉnh là `subagents`.
- **Tối ưu Token:** Context khổng lồ từ các công việc phụ sẽ được dịch vụ hóa thành chỉ duy nhất 1 báo cáo kết quả tóm tắt. Cần lưu ý Subagents là "stateless" (không có trạng thái), chúng không thể gửi rất nhiều tin nhắn trả ngược về agent cha trong phiên.

## 5. Quản Lý Ngữ Cảnh (Context management)
DeepAgents giải quyết bài toán giới hạn token thông qua:
- **Input context:** Được định hình ban đầu qua System prompt, memory, skills, và prompt của tools.
- **Compression (Nén):** Tự động tóm tắt và offload dữ liệu khi công việc kéo dài nhằm giữ context luôn nằm trong ngưỡng an toàn.
- **Isolation (Cách ly):** Sử dụng các Subagent để xử lý phần việc ngốn nhiều token, chỉ giữ lại kết quả (xem mục #4).
- **Long-term memory (Bộ nhớ dài hạn):** Lưu trữ cố định xuyên suốt nhiều thread qua hệ thống tập tin ảo.

## 6. Thực Thi Mã Nguồn (Code execution)
DeepAgents có thể chạy code bằng cách sử dụng các Backend hộp cát (sandbox).
- Khi sử dụng sandbox backend chuẩn `SandboxBackendProtocolV2`, agent tự động được cấp thêm công cụ `execute`.
- Nếu không có Sandbox backend, Agent hoàn toàn **không** có quyền dùng shell/chạy lệnh, chỉ giới hạn thao tác file văn bản bình thường.
- `execute` chạy lệnh và gộp cả stdout/stderr, trả về exit code, có cơ chế ngắt/lưu đầu ra quá dài thành tệp để agent tự thẩm định từ từ.
- **Sự quan trọng:** Cách ly chạy code mang tính bảo mật cho máy chủ gốc của bạn, môi trường sạch dễ dàng lặp lại.

## 7. Sự Tham Gia Của Con Người (Human-in-the-loop)
DeepAgents cho phép con người kiểm duyệt trước các hành vi của đặc vụ:
- Tham số cấu hình: `interrupt_on`.
- Truyền vào `create_deep_agent` dạng map. Ví dụ: `interrupt_on={"edit_file": True}` (dừng lại hỏi khi muốn sửa tệp).
- Tạm dừng thực thi trước khi thực hiện các yêu cầu phá hủy hoặc tốn tiền API.
- Người dùng có cơ hội phê duyệt, kiểm tra lại, hoặc chỉnh sửa input mà tool định dùng.

## 8. Kỹ Năng (Skills)
Mở rộng chức năng bằng chuẩn **Agent Skills**:
- Mỗi skill là một thư mục chứa tệp `SKILL.md` bao gồm thông tin metadata.
- **Tiết lộ Lũy tiến (Progressive disclosure):** Kỹ năng sẽ chỉ tải (load) vào khi framework xác định là context hiện tại rất hữu ích / cần đến nó. Agent chỉ đọc cái tóm tắt ở frontmatter trước, nếu cần mới bung cả file đọc cụ thể -> Điều này giúp tối ưu token.
- Gom cụm các chức năng xử lý, có thể kèm theo scripts, test templates giúp làm ra một agent linh hoạt mô-đun hóa cực kì mạnh.

## 9. Bộ Nhớ Dài Hạn (Memory)
DeepAgents lưu trí nhớ vượt cấp các session chat:
- Sử dụng chuẩn tệp `AGENTS.md` (persistent context).
- Trái ngược với Skills, Memory file sẽ **luôn luôn tải vào ngữ cảnh** khi bắt đầu khởi tạo.
- Khởi tạo agent với tham số `memory` trỏ vào đường dẫn file.
- Agent có thể trực tiếp tự **cập nhật lại Memory** sau khi tương tác với người dùng ở phiên hiện tại để rút kinh nghiệm cho tương lai.
- Rất dùng cho: lưu sở thích người dùng, hướng dẫn quy định dự án chung, kiến thức Domain cụ thể.
