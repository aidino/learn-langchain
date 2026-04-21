# Tổng quan về Frontend trong DeepAgents (Frontend Overview)

Tài liệu này hướng dẫn cách xây dựng giao diện người dùng (UI) có khả năng hiển thị luồng nội dung (streams) liên tục của subagent, quá trình thực hiện tác vụ (task progress) và hộp cát (sandbox) cho Deep Agents. Đây là giải pháp hữu hiệu khi tích hợp `deepagents==0.5.3` với các frontend hiện đại.

## Kiến trúc (Architecture)

Bằng cách sử dụng thư viện `@langchain/react`, bạn có thể dễ dàng kết nối ứng dụng React của mình với DeepAgent backend thông qua hook `useStream`. Hook này cho phép bạn không chỉ nhận streaming tin nhắn văn bản thông thường, mà còn có thể subscribe vào trạng thái nội bộ của Agent.

### React Hook `useStream`

Ví dụ dưới đây minh họa cách bắt luồng dữ liệu tiến trình công việc (`todos`) và trạng thái của các `subagents` để theo dõi theo thời gian thực:

```tsx
import { useStream } from "@langchain/react";

function App() {
  const stream = useStream<typeof agent>({
    apiUrl: "http://localhost:2024", // URL backend DeepAgents của bạn
    assistantId: "agent",
  });

  // Trích xuất các trạng thái sâu (deep agent state) bên ngoài định dạng message
  // Hiển thị tiến trình công việc dưới dạng To-Do list
  const todos = stream.values?.todos; 
  // Trích xuất thông tin subagents để biết tác vụ nào đang chạy ngầm
  const subagents = stream.subagents; 
  
  return (
    <div>
      {/* Hiển thị tiến trình công việc (Task Progress) */}
      {todos && <TaskList items={todos} />}
      
      {/* Trực quan hóa Subagents đang suy nghĩ */}
      {subagents && <SubAgentMonitor agents={subagents} />}
    </div>
  );
}
```

## Best Practices & Ví dụ thực tiễn (Context7 cho bản `0.5.3`)

Dựa trên các pattern từ Context7, dưới đây là các khuyến nghị hàng đầu khi phát triển Frontend dành riêng cho Deep Agents phiên bản 0.5.3:

### 1. Sử dụng ứng dụng `deep-agents-ui` làm Template xuất phát

Thay vì xây dựng UI hỗ trợ Sandbox, Human-in-the-loop hay Multi-agent streams từ đầu, bạn nên tận dụng framework `deep-agents-ui` của LangChain. Nó cung cấp sẵn bộ tính năng UI để quan sát và thao tác với Deep Agents:

```bash
# Clone mã nguồn deep-agents-ui
git clone https://github.com/langchain-ai/deep-agents-ui.git
cd deep-agents-ui

# Cài đặt dependency và chạy máy chủ phát triển
yarn install
yarn dev
```

### 2. Giao tiếp tiêu chuẩn hóa qua Protocols

Các Agent endpoint của bạn (được tạo bằng hàm `create_deep_agent`) đã được tích hợp sẵn các API đạt chuẩn, giúp Frontend dễ dàng móc nối dữ liệu:

- **Agent Protocol:** API tiêu chuẩn hóa mà framework Frontend sử dụng (`useStream`) để gửi/nhận dữ liệu.
- **Human-in-the-loop:** Tạo các rào chắn (gates) yêu cầu nút phê duyệt từ con người trên UI đối với các hành động rủi ro.
- **A2A (Agent-to-Agent):** Cho phép Frontend trực quan hóa luồng dữ liệu khi Agent A điều phối Subagent B.
- **Streaming Tokens:** Backend hỗ trợ truyền mượt mà từng stream token để tạo hiệu ứng viết gõ trực tiếp thông qua vòng lặp `for chunk in agent.stream(...)`.

### 3. Đồng bộ Backend Streaming cho Subagents

Để Frontend có thể bắt được danh sách `subagents`, Backend của bạn cần khai báo metadata subagent một cách minh bạch:

```python
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o")

# Cấu hình DeepAgent expose subagents context ra ngoài cho Frontend
agent = create_deep_agent(
    model=model,
    system_prompt="You are a helpful assistant",
    subagents=[
        {
            "name": "researcher",
            "description": "Research assistant cho các nhiệm vụ phức tạp",
        }
    ],
)
```

## Các Mẫu thiết kế liên quan (Related patterns)

- **Subagent streaming:** Chia nhỏ luồng tin nhắn và phân định rõ tin nhắn từ tác nhân hệ thống (Main Agent) và các tác nhân con (Subagents), dựa vào cơ chế `filterSubagentMessages`.
- Sử dụng các [LangChain Frontend Patterns](https://docs.langchain.com/oss/python/langchain/frontend/overview) khi giải quyết các vấn đề truyền tải UI khác như Rendering Markdown, Xử lý form, vv.

---
**Nguồn tham khảo:** 
- [DeepAgents Frontend Overview - Docs by LangChain](https://docs.langchain.com/oss/python/deepagents/frontend/overview)
- Context7 Codebase (deepagents==0.5.3)
