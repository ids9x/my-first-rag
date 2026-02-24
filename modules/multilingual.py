"""
English Prompt Template

This module contains the prompt template for English-only RAG queries.
Previously supported multilingual features but now simplified for English only.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ── English Prompt Template ────────────────────────────────────

PROMPT_EN = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant specializing in nuclear regulatory documents.\n"
     "Answer the question based ONLY on the following context.\n"
     "If the context does not contain enough information, say so.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}"),
])


def get_prompt() -> ChatPromptTemplate:
    """Return the English prompt template."""
    return PROMPT_EN
