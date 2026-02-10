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
from modules.multilingual import get_prompt
from modules.reranking import get_reranker, RerankingRetriever, MMRRerankingRetriever
from config.settings import RETRIEVER_K, RERANKER_FETCH_K, MMR_FETCH_K, MMR_LAMBDA_MULT


def format_docs(docs):
    """Format retrieved documents into context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def run_basic(manager: VectorStoreManager):
    """Basic vector-only RAG with English prompt."""
    # Get reranker if enabled
    reranker = get_reranker()

    # Reranking + MMR diversity (new default workflow)
    if reranker:
        retriever = MMRRerankingRetriever(
            vector_store=manager.get_store(),
            reranker=reranker,
            fetch_k=MMR_FETCH_K,
            final_k=RETRIEVER_K,
            lambda_mult=MMR_LAMBDA_MULT,
        )
        print(f"🔄 Reranking + MMR diversity enabled (fetch_k={MMR_FETCH_K}, λ={MMR_LAMBDA_MULT})")
    else:
        # Fallback: standard vector retrieval
        retriever = manager.get_retriever(k=RETRIEVER_K)
        print("📊 Standard vector retrieval")

    prompt = get_prompt()
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

        print("   Thinking...")
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

    # Get reranker if enabled
    reranker = get_reranker()

    retriever = HybridRetriever(
        vector_retriever=manager.get_retriever(k=RETRIEVER_K * 2),
        bm25_index=bm25,
        k=RETRIEVER_K,
        reranker=reranker,  # Pass reranker to hybrid retriever
    )

    if reranker:
        print("🔄 Hybrid retrieval with reranking enabled")

    prompt = get_prompt()
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

    # Get reranker if enabled
    reranker = get_reranker()
    agent = build_agent(manager, bm25_index=bm25, reranker=reranker)

    if reranker:
        print("🔄 Agentic mode with reranking enabled")

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
