# Đưa Deep Agents vào môi trường Production (Sản xuất)

Bài viết này tổng hợp và giải thích chi tiết các cân nhắc cùng phương pháp thực hành từ tài liệu [Going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production) của LangChain để đưa Deep Agents vào môi trường thực tế một cách an toàn và tối ưu.

---

## 1. Môi trường Triển khai (LangSmith Deployments)
Cách tốt nhất để triển khai Deep Agents là sử dụng **LangSmith Deployments** – một nền tảng cơ sở hạ tầng được quản lý (managed infrastructure). 
Hệ thống này hỗ trợ sẵn các tính năng quan trọng như:
- Xác thực (Authentication)
- Webhooks để kết nối với các ứng dụng khác
- Chạy lịch trình định kỳ (Cron jobs)
- Cấu hình agent thông qua `langgraph.json` và triển khai trên CLI bằng lệnh `deepagents deploy`.

---

## 2. Các vấn đề cần lưu ý khi đưa lên Production

### A. Đa người dùng (Multi-tenancy)
Khi ứng dụng có nhiều người dùng sử dụng cùng lúc, hệ thống cần ngăn chặn việc lộ dữ liệu chéo:
- **Định danh & Truy cập (User Identity & Access Control):** Cần gắn nhãn (tag) tài nguyên với metadata quyền sở hữu (ví dụ: `owner: user_id`), đồng thời lọc tài nguyên chặt chẽ để trả về các truy vấn tương ứng đúng định danh.
- **RBAC (Role-Based Access Control):** Cấp quyền truy cập hệ thống chi tiết dựa theo vai trò của người dùng trong hệ thống/team.
- **Proxy Thông tin xác thực End-user:** Cấp quyền để agent thay mặt người dùng gọi các dịch vụ bên ngoài (vd: đọc GitHub cá nhân) thông qua mô hình Agent Auth. Tuyệt đối dùng proxy sandbox để quản lý secret keys, thay vì nạp trực tiếp vào file log để không làm lộ keys.

### B. Xử lý Bất đồng bộ (Async)
Để đáp ứng truy cập lớn (high-concurrency), bạn phải xây dựng các phương thức bất đồng bộ hoàn toàn để không gây ra Thread blocking (tắc nghẽn kịch bản).
- Ưu tiên dùng phiên bản **bất đồng bộ** của các hàm trong Middleware, như gõ đè `.abefore_agent` và `.aafter_agent`.
- Bất kì công việc tạo mạng (Network calling), thiết lập cơ sở dữ liệu nhớ, hay quản lý thư mục Sandbox (MCP servers) đều nên có từ khoá `await`.

### C. Độ bền bỉ và Tính liên tục (Durability)
- **Time Travel (Du hành thời gian):** Nhờ cơ chế LangGraph lưu lại lịch sử thay đổi (checkpoint) sau mỗi thao tác (step), bạn có thể tua lại và gỡ lỗi dễ dàng tại bất kì state nào trong khứ nếu AI mắc lỗi.
- **Human-in-the-loop (Interrupts):** Đối với các tác vụ rủi ro như giao dịch thanh toán, quá trình có thể tạm dừng (*pause*) dài hạn (vài phút đến vài ngày) để đợi con người xác nhận trước khi tiếp diễn.

### D. Quản lý Bộ nhớ (Memory)
Quy định phạm vi bộ nhớ mà Deep Agent được phép phân vùng (Scoping):
- **Phạm vi Cuộc hội thoại (Thread):** Kiến thức chỉ lưu ngắn hạn trong 1 hội thoại hiện hành (Mặc định).
- **Phạm vi Người dùng (User):** Đặc biệt hữu hiệu. Agent có thể lưu lại những tùy chọn hoặc thông tin của một người dùng, sau đó đọc lại vào đầu hội thoại của các phiên khác của người dùng đấy để tạo độ cá nhân hoá (tạo bằng `StoreBackend`).
- **Phạm vi Trợ lý (Assistant) / Tổ chức (Organization):** Toàn bộ người dùng trong công ty sẽ dùng chung một bối cảnh kiến thức tập thể.

### E. Môi trường Thực thi (Execution Environment)
- **Hệ thống tệp - Filesystem:** Dùng cho thao tác đọc/ghi file thông thường.
  - *StateBackend*: Vùng nhớ tạm dùng 1 lần (sẽ dọn dẹp khi hết Thread).
  - *StoreBackend*: Vùng nhớ lâu dài phân theo Namespace.
  - *CompositeBackend*: Kết hợp cả hai loại để điều tiết linh hoạt.
- **Sandboxes (Hộp Cát):** Môi trường cách ly cần thiết khi Agent yêu cầu quyền **thực thi Code** (chạy Python/Shell) qua công cụ `execute`.
  - Có thể nạp mã script vào hộp cát trước khi Agent chạy với `upload_files()`. Và tải kết quả trả về bằng `download_files()` khi ngừng tiến trình thông qua Middleware.
  - Sandbox có vòng đời TTL giới hạn để giải phóng tự động khi Agent ngừng hoạt động.

### F. Rào chắn an toàn (Guardrails)
Trong production, để tránh trường hợp Agent vận hành ngoài tầm kiểm soát, tài liệu khuyên dùng Middleware bọc ngoài:
- **Rate limiting (Giới hạn tốc độ):** Áp dụng `ModelCallLimitMiddleware` hoặc `ToolCallLimitMiddleware` nhằm chống việc LLM tạo ra các vòng lặp câu lệnh vô hạn, dễ làm hệ thống cháy túi vì phí API.
- **Handling errors (Quản lý lỗi mượt mà):** 
  - `ModelRetryMiddleware`: Tự động thử lại khi API Timeout hoặc giới hạn hạn ngạch.
  - `ModelFallbackMiddleware`: Dùng model phụ (ví dụ: gpt-4o xuống dòng gpt-4o-mini hoặc Claude) để vá khi API model chính bị sập hoàn toàn.
  - `ToolRetryMiddleware`: Khi các tool ngoại vi lỗi kết nối.
- **Data Privacy (Bảo mật thông tin - PII):** Xóa các thông tin định danh cá nhân trước khi truyền câu lệnh (Prompt) đến LLM. Dùng `PIIMiddleware("email", strategy="redact")` sẽ mã hoá email thành [REDACTED_EMAIL], thẻ VISA đổi thành ****-1234 nhằm tránh công ty LLM lưu giữ thông tin quan trọng của khách hàng. 

### G. Giao diện (Frontend)
Kết nối giao diện React/Web client của bạn vào backend LangSmith thông qua SDK `@langchain/react` bằng hook `useStream`.
- Cấu hình quan trọng cần bật: `reconnectOnMount: true` dể Agent có thể trích xuất lại các tin nhắn khi web bị F5.
- Cấu hình `fetchStateHistory: true` cho phép web gọi ra toàn bộ Thread cũ cho lần xem sau.
