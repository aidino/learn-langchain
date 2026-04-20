import os
from typing import Literal
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo client cho Tavily
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Thực hiện tìm kiếm trên web"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )