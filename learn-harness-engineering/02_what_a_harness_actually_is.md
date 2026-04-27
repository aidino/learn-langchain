# Bài 2: Harness Thực Sự Là Gì? (What a Harness Actually Is)

Thuật ngữ "harness" thường bị hiểu nhầm trong lĩnh vực AI coding agent là chỉ "một file prompt". Nếu chỉ có file prompt thì chẳng khác nào mở nhà hàng mà chỉ có nguyên liệu — không có bếp, không dao, không công thức, không quy trình. Đó không phải là nhà hàng, đó là cái tủ lạnh.

Bài học này định nghĩa "harness" một cách chính xác và mang tính hành động cao: một harness bao gồm 5 hệ thống phụ (subsystems), mỗi hệ thống có trách nhiệm và tiêu chí đánh giá rõ ràng.

## Bắt Đầu Với Một Sự So Sánh
Hãy tưởng tượng bạn là một kỹ sư mới được giao vào một dự án không có tài liệu nào. Không có README, không comment trong code, không ai chỉ bạn cách chạy test, và cấu hình CI thì bị giấu ở đâu đó. Bạn có viết code tốt được không? Có thể — nhưng bạn sẽ mất vô số thời gian để "tìm hiểu xem dự án này nói về cái gì" thay vì "giải quyết vấn đề".

Một AI agent cũng đối mặt với tình huống y hệt, thậm chí tệ hơn vì nó không thể hỏi đồng nghiệp. Agent chỉ thấy những file bạn cung cấp và lệnh nó có thể chạy.

OpenAI cho rằng "repo CHÍNH LÀ spec (đặc tả)" — mọi context cần thiết phải nằm trong kho lưu trữ, được cung cấp qua các file hướng dẫn có cấu trúc, các lệnh xác minh rõ ràng và tổ chức thư mục gọn gàng.
Anthropic lại nhấn mạnh vào quản lý trạng thái, khả năng phục hồi rõ ràng và theo dõi tiến độ cho các agent chạy tác vụ dài.
Dù khác nhau, cả hai công ty đều chung quan điểm: **mọi thứ trong hạ tầng kỹ thuật nằm ngoài bản thân model AI sẽ quyết định xem khả năng của model đó được phát huy đến đâu.**

Hãy nhìn vào các công cụ hiện tại:
- **Claude Code**: Áp dụng tư duy harness. Nó đọc `CLAUDE.md` (kệ công thức), chạy shell command (kệ dao), thực thi trong môi trường local (bếp), duy trì lịch sử phiên (trạm chuẩn bị), và chạy test để xem kết quả (cửa sổ kiểm tra chất lượng). Nếu không có hướng dẫn chạy test, bước kiểm tra chất lượng sẽ bị hỏng.
- **Cursor**: `cursorrules` là kệ công thức, terminal là kệ dao. Tuy nhiên, khả năng quản lý trạng thái của Cursor khá yếu — đóng IDE rồi mở lại là mất context cũ.
- **AutoGPT**: Là một bài học đắt giá. Thiếu quản lý trạng thái có cấu trúc và thiếu cơ chế phản hồi chính xác khiến agent bị kẹt trong vòng lặp. AutoGPT không hiệu quả là do harness của nó bị hỏng.

## Các Khái Niệm Cốt Lõi (Core Concepts)
- **Harness là gì**: Mọi thứ thuộc về hạ tầng kỹ thuật nằm ngoài trọng số (weights) của model. OpenAI tóm gọn công việc của kỹ sư là: thiết kế môi trường, thể hiện ý định, và xây dựng vòng lặp phản hồi.
- **Repo là nguồn sự thật duy nhất (Single Source of Truth)**: Những gì agent không thấy thì coi như không tồn tại. Mọi context cần thiết phải nằm trong repo (qua cấu trúc thư mục rõ ràng và các file cấu trúc).
- **Cung cấp bản đồ, đừng đưa sách hướng dẫn**: File `AGENTS.md` nên giống một trang chỉ dẫn thư mục thay vì bách khoa toàn thư. Chỉ khoảng 100 dòng là đủ, nếu dài hơn hãy chia nhỏ vào thư mục `docs/`.
- **Ràng buộc, đừng quản lý vi mô (Constrain, don't micromanage)**: Sử dụng các quy tắc có thể thực thi để giới hạn agent thay vì liệt kê từng hướng dẫn một. Tách biệt giữa "người làm việc" và "người kiểm tra công việc" vì agent thường có xu hướng tự khen ngợi kết quả của mình.
- **Thử nghiệm loại bỏ từng thành phần (Isometric model control)**: Để đánh giá giá trị của một thành phần harness, hãy loại bỏ nó và xem điều gì khiến hiệu suất giảm mạnh nhất.

## Mô Hình Harness 5 Hệ Thống (The Five-Subsystem Harness Model)
Quay lại ví dụ nhà bếp, một harness hoàn chỉnh bao gồm 5 hệ thống chức năng:
1. **Hệ thống Hướng dẫn - Instruction subsystem (Kệ công thức)**: Tạo file `AGENTS.md` (hoặc `CLAUDE.md`) chứa tổng quan dự án, tech stack và phiên bản, các lệnh chạy thử lần đầu (như `make setup`, `make test`), các ràng buộc cứng không thể thương lượng (vd: "Tất cả API phải dùng OAuth 2.0"), và các link dẫn đến tài liệu chi tiết.
2. **Hệ thống Công cụ - Tool subsystem (Kệ dao)**: Đảm bảo agent có đủ quyền truy cập các công cụ. Đừng vô hiệu hóa shell vì lý do "bảo mật", nhưng cũng đừng mở tất cả mọi thứ — tuân theo nguyên tắc quyền tối thiểu (least-privilege).
3. **Hệ thống Môi trường - Environment subsystem (Bếp)**: Làm cho trạng thái môi trường tự mô tả được. Sử dụng `pyproject.toml` hoặc `package.json` để chốt phiên bản dependencies, `.nvmrc` hoặc `.python-version` cho phiên bản runtime, dùng Docker hoặc devcontainers để đảm bảo khả năng tái tạo.
4. **Hệ thống Trạng thái - State subsystem (Trạm chuẩn bị)**: Các tác vụ dài cần theo dõi tiến trình. Sử dụng file `PROGRESS.md` đơn giản để ghi lại: những gì đã làm, những gì đang làm, những gì đang bị chặn (blocked). Cập nhật trước khi kết thúc một phiên làm việc, đọc khi bắt đầu phiên mới.
5. **Hệ thống Phản hồi - Feedback subsystem (Cửa sổ kiểm tra chất lượng)**: Đây là hệ thống mang lại ROI (lợi tức đầu tư) cao nhất. Hãy liệt kê rõ ràng các lệnh xác minh (verification commands) trong `AGENTS.md` (ví dụ: lệnh chạy test, type check, lint). Nếu thiếu hệ thống này, bạn vẫn có thể code nhưng sẽ luôn loạng choạng.

## Câu Chuyện Thực Tế Của Một Đội Ngũ
Một nhóm sử dụng GPT-4o cho dự án frontend TypeScript + React (~20.000 dòng code). Họ trải qua 4 giai đoạn cải thiện harness:
- **Giai đoạn 1 — Bếp trống**: Chỉ có mô tả dự án cơ bản trong README. Tỷ lệ thành công chỉ đạt **20%** (chọn sai package manager, không theo quy tắc đặt tên, không thể chạy test).
- **Giai đoạn 2 — Thêm kệ công thức**: Thêm `AGENTS.md` mô tả tech stack, quy tắc đặt tên, kiến trúc chính. Tỷ lệ thành công tăng lên **60%**.
- **Giai đoạn 3 — Mở cửa sổ kiểm tra chất lượng**: Thêm các lệnh xác minh (test, lint, build) vào `AGENTS.md`. Tỷ lệ thành công tăng lên **80%**.
- **Giai đoạn 4 — Sẵn sàng trạm chuẩn bị**: Áp dụng các file theo dõi tiến độ (`PROGRESS.md`). Tỷ lệ thành công ổn định ở mức **80-100%**.

Kết quả: Chỉ với việc tổ chức lại "bếp" (harness), mà không cần đổi model, tỷ lệ thành công đã tăng từ 20% lên gần 100%.

## Bài Học Trọng Tâm (Key Takeaways)
- **Harness = Hướng dẫn (Instructions) + Công cụ (Tools) + Môi trường (Environment) + Trạng thái (State) + Phản hồi (Feedback).** Cả 5 hệ thống này đều thiết yếu.
- Nếu không phải là trọng số model (weights), thì đó chính là harness. Harness quyết định bao nhiêu khả năng của model được hiện thực hóa.
- **Hệ thống Phản hồi (Feedback)** thường có đầu tư thấp nhất nhưng lợi nhuận cao nhất. Hãy cung cấp các lệnh xác minh đúng chuẩn trước tiên.
- Sử dụng phương pháp loại trừ (isometric model control) để định lượng sự đóng góp của từng hệ thống phụ, đừng chỉ dựa vào cảm tính.
- Harness cũng sẽ bị hao mòn ("rot") giống như code. Hãy kiểm toán (audit) thường xuyên và trả "nợ harness" (harness debt) giống như cách bạn trả nợ kỹ thuật (technical debt).
