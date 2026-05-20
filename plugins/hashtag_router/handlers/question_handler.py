# plugins/hashtag_router/handlers/_http_client.py
"""Cliente HTTP compartilhado entre handlers que fazem POST pro servidor."""
from __future__ import annotations
import logging
import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_RESPONSE_CHARS = 3500


def post_query(endpoint: str, query: str, label: str = "query") -> str:
    """
    POST {endpoint} com body {"query": query}.
    Retorna 'answer' do JSON ou fallback pra resposta crua.
    """
    if not query:
        return f"❌ {label}: payload vazio. Use: #{label} <pergunta>"
    
    try:
        resp = requests.post(
            endpoint,
            json={"query": query},
            headers={"Content-Type": "application/json", "User-Agent": "Hermes/1.0"},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.Timeout:
        return f"⏱️ {label}: timeout em {DEFAULT_TIMEOUT}s"
    except requests.ConnectionError as e:
        return f"🔌 {label}: falha de conexão — {e}"
    except Exception as e:
        logger.exception(f"[{label}] erro inesperado")
        return f"❌ {label}: {type(e).__name__}"
    
    if resp.status_code >= 400:
        return f"❌ {label}: HTTP {resp.status_code}\n{resp.text[:300]}"
    
    # Tenta JSON com chave 'answer', cai pra texto cru
    try:
        data = resp.json()
        answer = data.get("answer") or data.get("response") or str(data)
    except ValueError:
        answer = resp.text
    
    if len(answer) > MAX_RESPONSE_CHARS:
        answer = answer[:MAX_RESPONSE_CHARS] + "\n\n... (truncado)"
    
    return answer
