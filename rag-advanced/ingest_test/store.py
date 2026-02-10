"""ChromaDB vector store for ingested chunks."""
import chromadb
from chromadb.config import Settings

from . import config


def get_collection():
    """Return (or create) the ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(records: list, embeddings: list[list[float]]):
    """
    Store chunk records with pre-computed embeddings.

    records: list of ChunkRecord
    embeddings: list of embedding vectors (one per record)
    """
    collection = get_collection()
    collection.upsert(
        ids=[r.id for r in records],
        documents=[r.contextualized_text for r in records],
        embeddings=embeddings,
        metadatas=[r.metadata for r in records],
    )
    return len(records)


def query_store(query_embedding: list[float], top_k: int = None,
                where_filter: dict = None) -> dict:
    """
    Query the store with a pre-computed embedding.

    where_filter example: {"doc_type": "requirement"}
    """
    collection = get_collection()
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k or config.TOP_K,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter
    return collection.query(**kwargs)


def reset_store():
    """Delete and recreate the collection."""
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except ValueError:
        pass
    return get_collection()