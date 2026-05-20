# plugins/hashtag_router/handlers/leo_query.py
import os
from ._http_client import post_query

# TODO: substituir pela URL real, ou setar via .env
LEO_QUERY_URL = os.getenv("LEO_QUERY_URL", "https://leo-query.swiftp.com.br/api/query")


def handle_leo_query(query: str, user: str) -> str:
    return post_query(LEO_QUERY_URL, query, label="leo-query")
