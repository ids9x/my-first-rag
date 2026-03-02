"""
Ingest documents into the vector store.

Supports PDF, DOCX, XLSX, Email (.eml/.msg), and TXT files.

Usage:
    python -m scripts.ingest                          # Ingest all documents in data/
    python -m scripts.ingest --strategy section        # Use section-aware chunking
    python -m scripts.ingest --force                   # Re-ingest everything
    python -m scripts.ingest --build-bm25              # Also build BM25 index
    python -m scripts.ingest --build-kg                # Also extract knowledge graph
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.documents import Document
from core.store import VectorStoreManager
from modules.chunking import load_directory
from modules.hybrid_search import BM25Index
from modules.knowledge_graph import KnowledgeGraph
from config.settings import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Ingest documents (PDF, DOCX, XLSX, Email, TXT) into the RAG pipeline")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "section"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of all documents",
    )
    parser.add_argument(
        "--build-bm25",
        action="store_true",
        help="Also build/rebuild the BM25 keyword index",
    )
    parser.add_argument(
        "--build-kg",
        action="store_true",
        help="Also extract knowledge graph triples (slow!)",
    )
    args = parser.parse_args()

    # Initialize the vector store manager
    manager = VectorStoreManager()

    if args.force:
        manager.reset()

    # Check what's already ingested
    skip = set()
    if not args.force and manager.exists:
        skip = manager.get_source_files()
        if skip:
            print(f"📋 Already ingested: {', '.join(skip)}")

    # Load and chunk PDFs
    chunks = load_directory(DATA_DIR, strategy=args.strategy, skip_existing=skip)

    if not chunks:
        if skip:
            print("✅ All documents already ingested. Use --force to re-ingest.")
        else:
            print(f"❌ No supported documents found in {DATA_DIR}. Add files and try again.")
        return

    # Add to vector store
    manager.add_documents(chunks)

    # Optionally build BM25 index
    if args.build_bm25:
        print("\n🔤 Building BM25 keyword index...")
        # Need ALL chunks for BM25, not just new ones
        # Fetch in batches to avoid SQLite variable limit
        store = manager.get_store()
        collection = store._collection
        total = collection.count()
        batch_size = 5000
        all_docs = []
        for offset in range(0, total, batch_size):
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for content, metadata in zip(
                batch.get("documents", []),
                batch.get("metadatas", []),
            ):
                all_docs.append(Document(page_content=content, metadata=metadata))
            print(f"   Loaded {min(offset + batch_size, total)}/{total} chunks...")

        bm25 = BM25Index(all_docs)
        bm25.save()

    # Optionally extract knowledge graph
    if args.build_kg:
        print("\n🕸️  Extracting knowledge graph (this may take a while)...")
        kg = KnowledgeGraph()
        kg.extract_from_chunks(chunks)
        print(f"   Graph stats: {kg.get_stats()}")

    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
