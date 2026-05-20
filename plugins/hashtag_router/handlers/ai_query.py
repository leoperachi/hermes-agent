# plugins/hashtag_router/handlers/ai_query.py
import os
from ._http_client import post_query

# TODO: substituir pela URL real, ou setar via .env
AI_QUERY_URL = os.getenv("AI_QUERY_URL", "https://ai-query.swiftp.com.br/api/query")


def handle_ai_query(query: str, user: str) -> str:
    return post_query(AI_QUERY_URL, query, label="ai-query")
