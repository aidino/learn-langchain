# Bài 1: Tại Sao Các Agent Mạnh Mẽ Vẫn Thất Bại? (Why Capable Agents Still Fail)

*(Dựa trên: Lecture 01. Strong Models Don't Mean Reliable Execution | Learn Harness Engineering)*

---

## 1. Vấn đề thực tế
Bạn có thể đang sử dụng những mô hình AI mạnh nhất thế giới (như Claude Pro, GPT-4o) nhưng khi giao phó một dự án thực tế, Agent (đại diện AI) vẫn thường xuyên:
- Thêm tính năng mới nhưng làm hỏng các test case cũ.
- Sửa một lỗi nhưng sinh ra hai lỗi khác.
- Chạy suốt 20 phút rồi tự tin báo cáo "đã xong", nhưng code viết ra lại sai hoàn toàn so với yêu cầu.

**Phản xạ đầu tiên:** *"Mô hình này chưa đủ tốt, cần nâng cấp mô hình đắt tiền hơn!"*
**Sự thật:** Nguyên nhân thường không nằm ở bản thân mô hình. Ngay cả các agent lập trình mạnh nhất trên hệ thống đánh giá SWE-bench cũng chỉ đạt tỷ lệ thành công khoảng 50-60%. Trong môi trường làm việc thực tế với các yêu cầu mơ hồ và quy ước ngầm, con số này còn thấp hơn rất nhiều.

---

## 2. Thí nghiệm "Cùng một con ngựa, khác số phận"
Anthropic đã làm một thí nghiệm: Yêu cầu AI làm một tựa game 2D retro với cùng một mô hình (Opus 4.5) và cùng một prompt.
- **Lần 1 (Không có hỗ trợ):** Mất 20 phút, tốn 9$, các tính năng cốt lõi của game hoàn toàn không hoạt động.
- **Lần 2 (Có Harness đầy đủ - kiến trúc 3 agent: planner + generator + evaluator):** Mất 6 giờ, tốn 200$, game **chơi được**.

**Kết luận:** Bản thân mô hình (con ngựa) không thay đổi. Thứ thay đổi là **Harness** (yên ngựa) – tức là toàn bộ cơ sở hạ tầng kỹ thuật và môi trường xung quanh hỗ trợ cho mô hình. Mã nguồn dù mạnh đến đâu mà không có "yên ngựa" tốt thì cũng khó đi xa.

---

## 3. Tại sao các Agent thực sự thất bại?
Có 5 "cạm bẫy" chính khiến AI làm việc không hiệu quả:

1. **Yêu cầu (Task) không rõ ràng:** Khi bạn yêu cầu "thêm tính năng tìm kiếm", AI phải tự đoán xem bạn muốn tìm kiếm toàn văn hay theo cấu trúc, có phân trang không... Nếu đoán sai, chi phí sửa chữa rất tốn kém.
2. **Thiếu quy ước kiến trúc ngầm:** Dự án của bạn có những quy ước riêng (ví dụ: dùng SQLAlchemy 2.0, API phải có OAuth 2.0) nhưng những điều này chỉ nằm trong đầu lập trình viên. AI không biết những luật lệ này và mặc định viết theo cách chung chung.
3. **Môi trường thực thi (Environment) có vấn đề:** Thiếu thư viện, sai phiên bản tool (Node, Python). Agent sẽ lãng phí "bộ nhớ ngữ cảnh" (context window) để sửa lỗi `pip install` thay vì làm nhiệm vụ chính.
4. **Không có cơ chế xác minh (Verification):** Không có test, không có linter, không có lệnh kiểm tra nào được cung cấp. Agent viết code xong, tự nhìn thấy có vẻ ổn liền báo cáo "đã xong". (Giống như làm bài tập mà không có đáp án để tự chấm).
5. **Nhiệm vụ kéo dài bị mất tính liên tục:** Đối với các phiên làm việc dài (trên 30 phút), Agent thường quên mất những gì đã khám phá ở phiên trước đó, dẫn đến việc phải làm lại từ đầu.

---

## 4. Các khái niệm cốt lõi cần nhớ
- **Harness (Bộ khung/Yên ngựa):** Mọi thứ bên ngoài trọng số mô hình (hướng dẫn, công cụ, môi trường, quản lý trạng thái, phản hồi kiểm thử).
- **Harness-Induced Failure (Thất bại do Harness):** Mô hình có đủ năng lực nhưng bị thất bại do môi trường thực thi có cấu trúc kém.
- **Verification Gap (Khoảng trống xác minh):** Khoảng cách giữa sự tự tin của Agent ("Tôi làm xong rồi") và sự chính xác thực tế.
- **Definition of Done (Định nghĩa hoàn thành):** Một tập hợp các điều kiện mà máy có thể tự xác minh (VD: test pass, type check pass, linting sạch).
- **Diagnostic Loop (Vòng lặp chẩn đoán):** Quy trình: Thực thi -> Thấy lỗi -> Quy lỗi cho một lớp cụ thể của Harness -> Sửa lớp đó -> Thực thi lại.

---

## 5. Giải pháp: Khi thất bại, hãy sửa Harness trước!
Nguyên tắc cốt lõi: Khi mọi thứ không hoạt động, **đừng vội đổi mô hình – hãy kiểm tra Harness.** 
Cách thực hiện:

1. **Quy trách nhiệm lỗi rõ ràng:** Đừng nói "mô hình ngu quá". Hãy hỏi: Yêu cầu có rõ ràng không? Có đủ ngữ cảnh chưa? Có phương pháp kiểm thử nào không?
2. **Viết "Definition of Done" thật cụ thể:** Thay vì "thêm tính năng tìm kiếm", hãy viết rõ tiêu chí hoàn thành như: 
   - *API GET `/api/search?q=xxx`*
   - *Hỗ trợ phân trang, mặc định 20 item.*
   - *Mọi code mới phải pass lệnh `pytest` và `mypy --strict`.*
3. **Tạo file `AGENTS.md` (Rất Quan Trọng):** Đặt file này ở thư mục gốc của dự án để chỉ cho Agent biết về tech stack, kiến trúc và các câu lệnh kiểm thử của dự án. **Một file `AGENTS.md` thậm chí còn mang lại hiệu quả cao hơn việc trả tiền nâng cấp lên mô hình xịn hơn.**
4. **Xây dựng Vòng lặp chẩn đoán (Diagnostic Loop):** Xem mỗi lỗi lầm của AI là một tín hiệu cho thấy Harness của bạn đang có lỗ hổng. Sửa lỗ hổng đó để những lần sau AI không vấp phải nữa.

---

## Tóm tắt (Key Takeaways)
- Năng lực của mô hình và sự đáng tin cậy khi thực thi là 2 chuyện khác nhau.
- Thay thế mô hình là cách giải quyết đắt đỏ nhất và thường không giải quyết đúng bản chất vấn đề.
- 5 lớp phòng thủ cần kiểm tra khi AI làm sai: Đặc tả nhiệm vụ, Cung cấp ngữ cảnh, Môi trường thực thi, Phản hồi xác minh, Quản lý trạng thái.
- **Hãy bắt đầu bằng việc tạo một file `AGENTS.md` cho dự án của bạn!**
