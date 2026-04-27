# Giới thiệu về Learn Harness Engineering

**Harness Engineering** (Kỹ thuật Điều khiển/Ràng buộc) là một lĩnh vực mới nổi tập trung vào việc thiết kế môi trường làm việc, quy trình và các bộ quy tắc nhằm giúp các AI Agent (như Claude Code, Codex) hoạt động một cách ổn định, đáng tin cậy và có thể dự đoán được.

Bài học mở đầu (Welcome) của trang web `learn-harness-engineering` đưa ra một cái nhìn tổng quan về khóa học này. Dưới đây là những nội dung chính:

## 1. Mục tiêu của khóa học
Khóa học tổng hợp những lý thuyết và thực tiễn tiên tiến nhất từ các tổ chức hàng đầu như **OpenAI** và **Anthropic**. Thay vì cố gắng làm cho mô hình AI "thông minh hơn", mục tiêu là xây dựng một hệ thống làm việc khép kín (closed-loop) giúp AI:
- Xây dựng tính năng và sửa lỗi hiệu quả.
- Tự động hóa các tác vụ phát triển phần mềm mà không bị lạc lối hay đưa ra những thay đổi ngoài ý muốn.

## 2. Cơ chế cốt lõi của một Harness (Bộ điều khiển)
Một Harness đóng vai trò như một bộ khung bao bọc quanh AI Agent. Nhiệm vụ của nó là:
- Đặt ra ranh giới và quy tắc rõ ràng để ràng buộc hành vi của AI.
- Quản lý trạng thái và duy trì ngữ cảnh khi AI thực hiện các tác vụ dài hạn (long-running tasks) qua nhiều phiên làm việc.
- Cung cấp cơ chế xác minh và kiểm soát để đảm bảo AI đang đi đúng hướng.

## 3. Những kỹ năng bạn sẽ học được
Khi tham gia khóa học, bạn sẽ nắm được cách:
- **Ràng buộc hành vi của Agent:** Sử dụng các quy tắc và ranh giới rõ ràng (ví dụ: qua file `AGENTS.md`).
- **Duy trì ngữ cảnh (Context):** Quản lý hiệu quả bộ nhớ và ngữ cảnh cho các tác vụ phức tạp, kéo dài.
- **Ngăn chặn AI "ảo tưởng" thành công:** Tránh tình trạng AI tuyên bố đã hoàn thành công việc (declare victory) khi thực tế vẫn còn lỗi hoặc chưa đạt yêu cầu.
- **Xác minh (Verification):** Thiết lập quy trình kiểm thử toàn diện (full-pipeline tests) và buộc AI tự đánh giá (self-reflection) kết quả công việc.
- **Tăng khả năng quan sát (Observability):** Làm cho quá trình AI hoạt động trở nên minh bạch và dễ dàng gỡ lỗi (debug) hơn.

## 4. Cấu trúc của khóa học
Khóa học được chia làm 3 phần chính:
1. **Lectures (Bài giảng lý thuyết):** Hiểu được lý do tại sao những AI mạnh mẽ vẫn có thể thất bại và lý thuyết đằng sau việc xây dựng một Harness hiệu quả.
2. **Projects (Dự án thực hành):** Thực hành tự tay xây dựng một môi trường Agentic đáng tin cậy từ con số không.
3. **Resource Library (Thư viện tài nguyên):** Cung cấp sẵn các biểu mẫu có thể sao chép và dùng ngay (như `AGENTS.md`, `feature_list.json`) cho dự án của riêng bạn.

## Các bước tiếp theo để học tập
- Đọc bài giảng đầu tiên: "Vì sao những Agent mạnh mẽ vẫn thất bại" (Why Capable Agents Still Fail).
- Bắt tay vào Project 01 để làm quen với tác vụ thực tế đầu tiên.
- Khám phá các mẫu (Templates) để áp dụng Harness vào dự án cá nhân.
