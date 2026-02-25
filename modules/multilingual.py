"""
Prompt Templates

Builds the RAG prompt template with a configurable system prompt.
Supports preset prompts (defined in config/settings.py) and
fully custom system prompts passed from the UI at query time.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.settings import PROMPT_PRESETS, DEFAULT_PROMPT_PRESET


def get_prompt(system_prompt: str | None = None) -> ChatPromptTemplate:
    """
    Return a prompt template with the given system prompt.

    Args:
        system_prompt: Custom system prompt text. If None or empty,
                       uses the default preset from config.

    Returns:
        ChatPromptTemplate with system, chat_history, and human messages.
    """
    if not system_prompt:
        system_prompt = PROMPT_PRESETS[DEFAULT_PROMPT_PRESET]

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{question}"),
    ])
