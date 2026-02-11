"""
Interactive query interface.

Usage:
    python -m scripts.query                   # Basic RAG (vector retrieval)
    python -m scripts.query --mode hybrid     # Hybrid search (vector + BM25)
    python -m scripts.query --mode agentic    # Agentic multi-step reasoning
    python -m scripts.query --mode kg         # Knowledge graph lookup
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.store import VectorStoreManager
from core.query_service import QueryService
from config.settings import RERANKER_ENABLED, MMR_FETCH_K, MMR_LAMBDA_MULT


def run_basic(manager: VectorStoreManager):
    """Basic vector-only RAG with English prompt."""
    service = QueryService(manager)

    # Show status message
    if RERANKER_ENABLED:
        print(f"🔄 Reranking + MMR diversity enabled (fetch_k={MMR_FETCH_K}, λ={MMR_LAMBDA_MULT})")
    else:
        print("📊 Standard vector retrieval")

    print("\n✅ Basic RAG ready. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("   Thinking...")
        try:
            result = service.query_basic(question)
            print(f"\nAssistant: {result['answer']}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_hybrid(manager: VectorStoreManager):
    """Hybrid vector + BM25 retrieval."""
    service = QueryService(manager)

    if not service.bm25_index:
        print("❌ No BM25 index found. Run: python -m scripts.ingest --build-bm25")
        return

    if RERANKER_ENABLED:
        print("🔄 Hybrid retrieval with reranking enabled")

    print("\n✅ Hybrid RAG ready (vector + BM25). Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("   Thinking (hybrid search)...")
        try:
            result = service.query_hybrid(question)
            print(f"\nAssistant: {result['answer']}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_agentic(manager: VectorStoreManager):
    """Agentic RAG with tool-calling."""
    service = QueryService(manager)

    if not service.bm25_index:
        print("⚠️  No BM25 index found — agent will only have semantic search.")

    if RERANKER_ENABLED:
        print("🔄 Agentic mode with reranking enabled")

    print("\n✅ Agentic RAG ready. The agent will show its reasoning. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print()
        try:
            result = service.query_agentic(question)
            print(f"\nAssistant: {result['answer']}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_kg():
    """Knowledge graph query mode."""
    from modules.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    if kg.graph.number_of_nodes() == 0:
        print("❌ Knowledge graph is empty. Run: python -m scripts.ingest --build-kg")
        return

    print(f"\n✅ Knowledge graph loaded. {kg.get_stats()}")
    print("   Enter an entity name to find its relationships. Type 'quit' to exit.\n")

    while True:
        entity = input("Entity: ").strip()
        if entity.lower() in ("quit", "exit", "q"):
            break
        if not entity:
            continue

        result = kg.format_query_results(entity)
        print(f"\n{result}\n")


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument(
        "--mode",
        choices=["basic", "hybrid", "agentic", "kg"],
        default="basic",
        help="Query mode (default: basic)",
    )
    args = parser.parse_args()

    if args.mode == "kg":
        run_kg()
        return

    # All other modes need the vector store
    manager = VectorStoreManager()
    if not manager.exists:
        print("❌ No vector store found. Run: python -m scripts.ingest")
        return

    if args.mode == "hybrid":
        run_hybrid(manager)
    elif args.mode == "agentic":
        run_agentic(manager)
    else:
        run_basic(manager)

    print("Goodbye!")


if __name__ == "__main__":
    main()
