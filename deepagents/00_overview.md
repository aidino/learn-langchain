# Tổng quan về Deep Agents (LangChain)

**Deep Agents** là một thư viện độc lập mã nguồn mở do LangChain phát triển, đóng vai trò như một "agent harness" (bộ khung tác nhân được cấu hình sẵn). Nó được xây dựng dựa trên các khối kiến trúc cốt lõi của LangChain và sử dụng **LangGraph** làm môi trường thực thi (runtime) để đảm bảo độ bền vững, khả năng xử lý luồng (streaming) và duy trì trạng thái (persistence).

Thay vì phải tự thiết lập prompt, công cụ và quản lý ngữ cảnh từ con số 0, Deep Agents cung cấp một mô hình tác nhân "batteries-included" (tích hợp sẵn mọi thứ cần thiết) có thể hoạt động ngay lập tức.

## 1. Các tính năng cốt lõi (Core Capabilities)

Deep Agents khắc phục nhược điểm "nông" (shallow) của các tác nhân AI thông thường bằng cách kết hợp các yếu tố then chốt giúp chúng xử lý các tác vụ phức tạp, kéo dài:

* **Lên kế hoạch và chia nhỏ tác vụ (Planning & Task Decomposition):** Tích hợp sẵn công cụ `write_todos` giúp tác nhân chia nhỏ một nhiệm vụ phức tạp thành nhiều bước, theo dõi tiến độ và điều chỉnh kế hoạch khi có thông tin mới.
* **Hệ thống tệp (Filesystem):** Tác nhân có khả năng đọc/ghi tệp qua các công cụ như `read_file`, `write_file`, `edit_file`, `ls`, `grep`... Điều này cho phép chúng quản lý ngữ cảnh tốt hơn bằng cách lưu trữ thông tin trên hệ thống thay vì nhồi nhét tất cả vào cửa sổ ngữ cảnh (context window).
* **Tác nhân phụ (Sub-agents):** Khả năng ủy quyền công việc cho các chuyên gia (tác nhân phụ) với không gian ngữ cảnh biệt lập. Điều này giúp xử lý tác vụ song song và tránh làm nhiễu thông tin của tác nhân chính.
* **Quản lý ngữ cảnh & Bộ nhớ thông minh:** Khả năng tự động nén/tóm tắt các cuộc hội thoại dài và chuyển các kết quả đầu ra lớn vào hệ thống lưu trữ để giải phóng bộ nhớ.
* **Truy cập Shell (Shell Access):** Hỗ trợ công cụ `execute` để chạy mã/lệnh trực tiếp (khuyến nghị dùng trong môi trường sandbox như LangSmith Sandbox).

## 2. Khi nào nên sử dụng Deep Agents? (So sánh với LangChain & LangGraph)

* **Deep Agents:** Lựa chọn hàng đầu khi bạn muốn xây dựng các tác nhân hoạt động dài hạn, tự chủ cao (như nghiên cứu sâu, lập trình) và muốn tận dụng ngay các tính năng có sẵn (hệ thống tệp, lên kế hoạch, sub-agents).
* **LangChain:** Sử dụng khi bạn muốn xây dựng vòng lặp tác nhân cốt lõi nhưng muốn tự thiết kế từ đầu (tự viết prompt, tự định nghĩa tool) cho các nhu cầu đơn giản hơn.
* **LangGraph:** Sử dụng khi bạn cần một hệ thống điều phối cấp thấp (low-level) để thiết kế các luồng công việc phức tạp, kết hợp giữa quy trình truyền thống (deterministic) và tác nhân (agentic).

## 3. Cấu trúc của Hệ sinh thái Deep Agents

* **Deep Agents SDK:** Gói thư viện Python (tương thích đa mô hình LLM) để xây dựng các tác nhân có khả năng xử lý mọi tác vụ.
* **Deep Agents CLI:** Một tác nhân lập trình chạy trên terminal (command line) được cấu hình sẵn.
* **ACP Integration:** Cổng kết nối Agent Client Protocol để tích hợp tác nhân vào các trình soạn thảo mã (như Zed).

## 4. Cài đặt và Bắt đầu nhanh (Quickstart)

**Cài đặt thư viện:**
```bash
pip install deepagents
# hoặc dùng uv như trong file README của bạn:
# uv add deepagents tavily-python python-dotenv
```

**Ví dụ mã nguồn (Python):**
```python
from deepagents import create_deep_agent

# Khởi tạo tác nhân với các cấu hình mặc định thông minh
agent = create_deep_agent()

# Giao việc cho tác nhân
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Nghiên cứu về LangGraph và viết một bản tóm tắt chi tiết."}
    ]
})

print(result)
```

> **Lưu ý:** Tác nhân này hoàn toàn không bị trói buộc với một nhà cung cấp LLM cụ thể (provider-agnostic). Bạn có thể sử dụng OpenAI, Anthropic, Google, hoặc các mô hình mã nguồn mở chạy local.