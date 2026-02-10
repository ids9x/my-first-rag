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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from core.store import VectorStoreManager
from core.llm import get_llm
from modules.multilingual import detect_language, get_bilingual_prompt
from config.settings import RETRIEVER_K


def format_docs(docs):
    """Format retrieved documents into context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def run_basic(manager: VectorStoreManager):
    """Basic vector-only RAG with bilingual prompt."""
    retriever = manager.get_retriever(k=RETRIEVER_K)
    prompt = get_bilingual_prompt()
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ Basic RAG ready. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        lang = detect_language(question)
        print(f"   [detected: {lang}] Thinking...")
        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")


def run_hybrid(manager: VectorStoreManager):
    """Hybrid vector + BM25 retrieval."""
    from modules.hybrid_search import HybridRetriever, BM25Index

    try:
        bm25 = BM25Index.load()
    except FileNotFoundError:
        print("❌ No BM25 index found. Run: python -m scripts.ingest --build-bm25")
        return

    retriever = HybridRetriever(
        vector_retriever=manager.get_retriever(k=RETRIEVER_K * 2),
        bm25_index=bm25,
        k=RETRIEVER_K,
    )

    prompt = get_bilingual_prompt()
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ Hybrid RAG ready (vector + BM25). Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("   Thinking (hybrid search)...")
        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")


def run_agentic(manager: VectorStoreManager):
    """Agentic RAG with tool-calling."""
    from modules.agentic import build_agent
    from modules.hybrid_search import BM25Index

    bm25 = None
    try:
        bm25 = BM25Index.load()
    except FileNotFoundError:
        print("⚠️  No BM25 index found — agent will only have semantic search.")

    agent = build_agent(manager, bm25_index=bm25)

    print("\n✅ Agentic RAG ready. The agent will show its reasoning. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print()
        result = agent.invoke({"input": question})
        print(f"\nAssistant: {result['output']}\n")


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
