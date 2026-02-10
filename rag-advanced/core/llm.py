"""
LLM wrapper. Single place to configure the chat model.
"""
from langchain_ollama import ChatOllama
from config.settings import CHAT_MODEL, OLLAMA_BASE_URL, AGENT_TEMPERATURE


def get_llm(
    model: str = CHAT_MODEL,
    temperature: float = AGENT_TEMPERATURE,
) -> ChatOllama:
    """Return a ChatOllama instance."""
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )
