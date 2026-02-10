"""
Embedding model management.
Wraps OllamaEmbeddings so the model name is configured in one place.
"""
from langchain_ollama import OllamaEmbeddings
from config.settings import EMBED_MODEL, OLLAMA_BASE_URL


def get_embeddings(model: str = EMBED_MODEL) -> OllamaEmbeddings:
    """Return an OllamaEmbeddings instance."""
    return OllamaEmbeddings(
        model=model,
        base_url=OLLAMA_BASE_URL,
    )
