"""
Hashtag router for WhatsApp messages.

Intercepts messages BEFORE the LLM (Likely a misspelling or similar tool). If the message starts with a known hashtag, it triggers the corresponding handler and returns the response directly. Otherwise, it lets the message proceed to the LLM.
"""
from __future__ import annotations
from typing import Optional, Callable
import logging

from .handlers.pm2_handler import handle_pm2
from .handlers.leo_query import handle_leo_query
from .handlers.ai_query import handle_ai_query

logger = logging.getLogger(__name__)

# Mapa de hashtag → handler.
# Ordem importa: hashtags mais específicas primeiro (ex: #leo-query antes de #leo).
HANDLERS: dict[str, Callable[[str, str], str]] = {
    "#pm2": handle_pm2,
    "#leo-query": handle_leo_query,
    "#ai-query": handle_ai_query,
}


def route(message: str, user: str = "unknown") -> Optional[str]:
    """
    Tenta rotear a mensagem por hashtag.
    
    Returns:
        str: resposta do handler, se uma hashtag foi reconhecida
        None: mensagem não tem hashtag, deixa o LLM processar normalmente
    """
    if not message:
        return None
    
    text = message.strip()
    
    for tag, handler in HANDLERS.items():
        if text.lower().startswith(tag):
            payload = text[len(tag):].strip()
            logger.info(f"[hashtag_router] {tag} disparado por {user} | payload: {payload[:80]!r}")
            try:
                return handler(payload, user)
            except Exception as e:
                logger.exception(f"[hashtag_router] erro no handler {tag}")
                return f"❌ Erro ao processar {tag}: {type(e).__name__}: {e}"
    
    return None  # sem hashtag → segue fluxo normal do LLM
