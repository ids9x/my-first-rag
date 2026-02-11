"""
English Prompt Template

This module contains the prompt template for English-only RAG queries.
Previously supported multilingual features but now simplified for English only.
"""
from langchain_core.prompts import ChatPromptTemplate


# ── English Prompt Template ────────────────────────────────────

PROMPT_EN = ChatPromptTemplate.from_template(
    """You are a helpful assistant specializing in nuclear regulatory documents.
Answer the question based ONLY on the following context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
)


def get_prompt() -> ChatPromptTemplate:
    """Return the English prompt template."""
    return PROMPT_EN
