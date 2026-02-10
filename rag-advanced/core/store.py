"""
Persistent Vector Store Manager (Priority 1)

Handles ChromaDB creation, loading, and updating.
Key feature: detects whether a store already exists on disk and loads it
instead of re-embedding everything.
"""
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.embeddings import get_embeddings
from config.settings import CHROMA_DIR, DEFAULT_COLLECTION


class VectorStoreManager:
    """
    Manages a persistent ChromaDB vector store.

    Usage:
        manager = VectorStoreManager()

        # First run: ingest documents
        manager.add_documents(chunks)

        # Subsequent runs: just load
        store = manager.get_store()
        retriever = store.as_retriever(search_kwargs={"k": 4})
    """

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embeddings = get_embeddings()
        self._store: Chroma | None = None

    @property
    def exists(self) -> bool:
        """Check if a ChromaDB already exists on disk."""
        chroma_sqlite = self.persist_dir / "chroma.sqlite3"
        return chroma_sqlite.exists()

    def get_store(self) -> Chroma:
        """Load existing store or create an empty one."""
        if self._store is None:
            if self.exists:
                print(f"📂 Loading existing vector store from {self.persist_dir}")
            else:
                print(f"🆕 Creating new vector store at {self.persist_dir}")
                self.persist_dir.mkdir(parents=True, exist_ok=True)

            self._store = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )

            count = self._store._collection.count()
            print(f"   Store contains {count} chunks.")

        return self._store

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the store (embeds and persists automatically)."""
        store = self.get_store()
        print(f"➕ Adding {len(documents)} chunks to the store...")
        store.add_documents(documents)
        count = store._collection.count()
        print(f"   Store now contains {count} total chunks.")

    def reset(self) -> None:
        """Delete the store and start fresh."""
        import shutil

        if self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)
            print(f"🗑️  Deleted vector store at {self.persist_dir}")
        self._store = None

    def get_retriever(self, k: int = 4, **kwargs):
        """Convenience: return a retriever from the store."""
        store = self.get_store()
        return store.as_retriever(search_kwargs={"k": k, **kwargs})

    def get_source_files(self) -> set[str]:
        """Return the set of source filenames already in the store."""
        store = self.get_store()
        results = store.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sources
