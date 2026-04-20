from deepagents import create_deep_agent
from tools import internet_search
from dotenv import load_dotenv
load_dotenv()

research_instructions = """Bạn là một chuyên gia nghiên cứu. Công việc của bạn là tiến hành nghiên cứu kỹ lưỡng và sau đó viết một báo cáo hoàn chỉnh.
Bạn có quyền truy cập vào công cụ tìm kiếm trên internet như là phương tiện chính để thu thập thông tin."""

# Tạo agent
agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview", # Chuỗi định danh mô hình (ví dụ dùng Claude 3.5 Sonnet)
    tools=[internet_search],
    system_prompt=research_instructions,
)

# Gọi agent với câu hỏi cụ thể, ví dụ: "langgraph là gì?"
result = agent.invoke({"messages": [{"role": "user", "content": "Tình hình chiến tranh IRAN tính đến ngày hôm nay?"}]})

# In ra câu trả lời cuối cùng
print(result["messages"][-1].content[0]['text'])