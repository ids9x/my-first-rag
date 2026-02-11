"""
Reset the vector store and optional indices.

Usage:
    python -m scripts.reset_store              # Reset vector store only
    python -m scripts.reset_store --all        # Reset everything (store + BM25 + KG)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.store import VectorStoreManager
from config.settings import BM25_INDEX_PATH, KNOWLEDGE_GRAPH_DIR


def main():
    parser = argparse.ArgumentParser(description="Reset RAG stores and indices")
    parser.add_argument("--all", action="store_true", help="Reset everything")
    args = parser.parse_args()

    # Always reset vector store
    manager = VectorStoreManager()
    manager.reset()

    if args.all:
        # BM25 index
        bm25_path = Path(BM25_INDEX_PATH)
        if bm25_path.exists():
            bm25_path.unlink()
            print(f"🗑️  Deleted BM25 index at {bm25_path}")

        # Knowledge graph
        kg_dir = Path(KNOWLEDGE_GRAPH_DIR)
        if kg_dir.exists():
            import shutil
            shutil.rmtree(kg_dir)
            print(f"🗑️  Deleted knowledge graph at {kg_dir}")

    print("✅ Reset complete. Run 'python -m scripts.ingest' to rebuild.")


if __name__ == "__main__":
    main()
