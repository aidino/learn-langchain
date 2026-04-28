# Tùy chỉnh Deep Agents trong LangChain

Tài liệu này giải thích các khái niệm quan trọng khi tuỳ chỉnh Deep Agents trong LangChain, kèm theo ví dụ minh hoạ dễ hiểu dựa trên kiến trúc của framework này.

## 1. Middleware (Phần mềm trung gian)
- **Khái niệm**: Middleware giống như các "lớp lọc" hoặc "mô-đun mở rộng" bổ sung thêm các khả năng (như quản lý file, lập kế hoạch, uỷ quyền) cho agent mà không làm thay đổi vòng lặp xử lý cốt lõi của nó. Khi agent suy nghĩ hoặc gọi công cụ, các thao tác này có thể được điều chỉnh thông qua middleware.
- **Ví dụ**: Giống như khi bạn mua một chiếc điện thoại (Agent gốc), bạn có thể gắn thêm ốp lưng có sạc dự phòng (Middleware 1) hoặc cài thêm phần mềm diệt virus tự động quét tin nhắn (Middleware 2). Điện thoại vẫn nghe gọi bình thường nhưng có thêm tính năng ngầm.
  - `TodoListMiddleware`: Cung cấp cho agent khả năng tự động chia nhỏ công việc thành các bước (To-do list).
  - `SummarizationMiddleware`: Tự động tóm tắt lịch sử hội thoại khi đoạn chat quá dài để tránh bị vượt quá giới hạn bộ nhớ của LLM.

## 2. Subagents (Tác nhân phụ / Đại lý phụ)
- **Khái niệm**: Đây là các "trợ lý chuyên môn" do agent chính tạo ra (thông qua `SubAgentMiddleware`) để uỷ quyền xử lý các công việc phức tạp. Mỗi subagent chạy trong một không gian bộ nhớ (context) hoàn toàn độc lập, có các công cụ (tools) và prompt (hướng dẫn) riêng. Sau khi làm xong, subagent chỉ báo cáo kết quả tóm tắt lại cho agent chính.
- **Ví dụ**: Agent chính đóng vai trò là một "Giám đốc dự án". Khi cần thiết kế một hình ảnh, Giám đốc không tự làm mà triệu hồi một Subagent "Nhân viên thiết kế" (được trang bị riêng công cụ vẽ). Nhân viên này làm xong chỉ gửi lại bản vẽ hoàn chỉnh cho Giám đốc, thay vì kể lể chi tiết từng bước vẽ, giúp "đầu óc" (context) của Giám đốc không bị quá tải.

## 3. Backends (Hệ thống file ảo)
- **Khái niệm**: Backends quy định cách thức và nơi chốn để agent lưu trữ các file, log hoặc thông tin bộ nhớ của nó.
- **Các loại phổ biến**:
  - `StateBackend` (Mặc định): Lưu tạm thời trong trạng thái của LangGraph, tắt ứng dụng là mất.
  - `FilesystemBackend`: Ánh xạ trực tiếp vào một thư mục thật trên máy tính của bạn (để agent có thể đọc/ghi file như một lập trình viên).
  - `Sandbox Backends`: Môi trường an toàn (sandbox, ví dụ qua Docker, Modal, Daytona) để agent có thể chạy thử các đoạn code mà không sợ làm hỏng hệ điều hành gốc của bạn.
- **Ví dụ**: Giống như việc bạn chọn nơi lưu bài tập: có thể viết trên bảng phấn (StateBackend - xoá là mất), lưu vào ổ cứng USB (FilesystemBackend - lưu vĩnh viễn), hoặc chạy thử một phần mềm lạ nghiệm trên máy ảo (Sandbox Backend - an toàn tuyệt đối).

## 4. Human-in-the-loop (Sự can thiệp của con người)
- **Khái niệm**: Tính năng này (HITL) cho phép con người trực tiếp tham gia vào vòng lặp ra quyết định của AI. Khi agent chuẩn bị gọi một công cụ nhạy cảm hoặc quan trọng, hệ thống sẽ tạm dừng và chờ con người xem xét. Bạn có thể Phê duyệt (Approve), Chỉnh sửa thông số (Edit), hoặc Từ chối (Reject).
- **Ví dụ**: Agent tự động soạn một email báo giá cho khách hàng dựa trên yêu cầu của bạn. Nhưng trước khi gọi lệnh "Gửi Email", nó sẽ hiển thị lên màn hình: *"Tôi chuẩn bị gửi email này, bạn có đồng ý không?"*. Bạn phát hiện ra giá bị sai, nên bạn chọn **Edit** để sửa giá lại, sau đó mới cho phép nó gửi đi.

## 5. Skills (Kỹ năng)
- **Khái niệm**: Skills là các logic, quy trình chuẩn, hoặc "kiến thức chuyên môn" được đóng gói lại và nạp vào agent thông qua `SkillsMiddleware`. Nó giúp định hình cách agent giải quyết một bài toán cụ thể mà bạn không cần phải nhồi nhét mọi thứ vào System Prompt ban đầu.
- **Ví dụ**: Bạn nạp Skill "Quy tắc lập trình Python của công ty" cho agent. Từ đó về sau, mọi đoạn code Python do nó viết ra đều tự động tuân thủ chuẩn thụt lề, cách đặt tên biến theo đúng ý bạn.

## 6. Memory (Bộ nhớ dài hạn)
- **Khái niệm**: Khác với bộ nhớ ngắn hạn của một đoạn chat, Memory là khả năng duy trì ngữ cảnh, ghi nhớ các sở thích, thói quen hoặc thông tin từ người dùng xuyên suốt nhiều phiên làm việc (sessions) khác nhau. Memory thường được lưu trữ bền vững trên ổ cứng hoặc cơ sở dữ liệu.
- **Ví dụ**: Hôm qua bạn bảo agent: *"Tôi không thích dùng thư viện Pandas, hãy luôn dùng Polars"*. Nó lưu thông tin này vào Memory. Ngày mai, dù bạn mở một đoạn chat hoàn toàn mới và yêu cầu *"Hãy phân tích file CSV này"*, nó sẽ tự động nhớ và dùng Polars để xử lý mà bạn không cần phải nhắc lại.

---

### Tóm tắt kiến trúc khi kết hợp:
Bạn có thể tưởng tượng Deep Agents là một công ty:
- **Agent chính** là Giám đốc.
- **Memory** là sổ tay ghi chép thói quen của sếp.
- **Skills** là sổ tay nghiệp vụ công ty.
- **Subagents** là các nhân viên cấp dưới.
- **Middleware** là các phòng ban hỗ trợ (phòng IT quản lý file, phòng kế hoạch).
- **Human-in-the-loop** là Hội đồng quản trị (Bạn) - luôn xét duyệt các quyết định lớn nhất.
- **Backends** là kho lưu trữ hồ sơ và phòng thí nghiệm (Sandbox) của công ty.
