"""
LLM wrapper. Single place to configure the chat model.
"""
from langchain_openai import ChatOpenAI
from config.settings import CHAT_MODEL, LLAMA_SERVER_BASE_URL, LLAMA_SERVER_API_KEY, AGENT_TEMPERATURE


def get_llm(
    model: str = CHAT_MODEL,
    temperature: float = AGENT_TEMPERATURE,
) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance configured for llama.cpp server.

    The llama-server provides an OpenAI-compatible API at /v1/chat/completions.
    This allows us to use ChatOpenAI instead of ChatOllama, avoiding the
    Ollama nil pointer bug with Qwen3 MoE models.

    Args:
        model: Model name (not used by llama-server but required by OpenAI client)
        temperature: Sampling temperature for generation

    Returns:
        ChatOpenAI instance connected to llama-server
    """
    return ChatOpenAI(
        model=model,
        base_url=LLAMA_SERVER_BASE_URL,
        api_key=LLAMA_SERVER_API_KEY,
        temperature=temperature,
        request_timeout=120,  # 2 minute timeout for large model
    )
