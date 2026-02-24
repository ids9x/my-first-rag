"""
Service layer for RAG queries.

Provides clean, reusable functions for Basic, Hybrid, and Agentic query modes.
Used by both CLI (scripts/query.py) and web UI (app.py).
"""
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from core.store import VectorStoreManager
from core.llm import get_llm
from modules.multilingual import get_prompt
from modules.reranking import get_reranker, MMRRerankingRetriever
from modules.hybrid_search import HybridRetriever, BM25Index
from modules.agentic import build_agent
from config.settings import (
    RETRIEVER_K,
    MMR_FETCH_K,
    MMR_LAMBDA_MULT,
    CHROMA_DIR,
    BM25_INDEX_PATH,
    RERANKER_ENABLED,
)


def format_docs(docs):
    """Format retrieved documents into context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class QueryService:
    """
    Service layer for RAG queries.

    Handles initialization, chain building, and query execution for all modes.
    Returns structured data instead of printing to console.
    """

    def __init__(self, manager: Optional[VectorStoreManager] = None):
        """
        Initialize the query service.

        Args:
            manager: Optional pre-initialized VectorStoreManager.
                    If None, will create and check for existing store.
        """
        self.manager = manager if manager else VectorStoreManager()
        self.reranker = get_reranker() if RERANKER_ENABLED else None

        # Load BM25 if available
        self.bm25_index = None
        try:
            self.bm25_index = BM25Index.load()
        except FileNotFoundError:
            pass

        # Cache chains to avoid rebuilding
        self._basic_chain = None
        self._hybrid_chain = None
        self._agent = None

    def get_status(self) -> dict:
        """
        Get system status.

        Returns:
            dict with vector_store_exists, vector_store_count,
            bm25_exists, reranker_enabled
        """
        chunk_count = 0
        if self.manager.exists:
            try:
                # Get collection info
                collection = self.manager.get_store()._collection
                chunk_count = collection.count()
            except:
                pass

        return {
            "vector_store_exists": self.manager.exists,
            "vector_store_count": chunk_count,
            "bm25_exists": self.bm25_index is not None,
            "reranker_enabled": self.reranker is not None,
        }

    def _get_retriever_with_reranking(self):
        """Get retriever with optional reranking."""
        if self.reranker:
            return MMRRerankingRetriever(
                vector_store=self.manager.get_store(),
                reranker=self.reranker,
                fetch_k=MMR_FETCH_K,
                final_k=RETRIEVER_K,
                lambda_mult=MMR_LAMBDA_MULT,
            )
        else:
            return self.manager.get_retriever(k=RETRIEVER_K)

    def _extract_sources(self, docs) -> list[dict]:
        """Extract source metadata from documents."""
        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source_file", doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", "?"),
                "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
            })
        return sources

    def _ensure_basic_chain(self):
        """Build basic chain if not cached."""
        if self._basic_chain is None:
            retriever = self._get_retriever_with_reranking()
            prompt = get_prompt()
            llm = get_llm()

            self._basic_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            self._basic_retriever = retriever

    def query_basic(self, question: str) -> dict:
        """
        Execute basic vector search query.

        Args:
            question: User question

        Returns:
            dict with:
                - answer: str
                - sources: list[dict]
                - mode: "basic"
        """
        if not self.manager.exists:
            raise ValueError("Vector store not found. Run: python -m scripts.ingest")

        self._ensure_basic_chain()

        # Execute query
        answer = self._basic_chain.invoke(question)

        # Get source documents
        docs = self._basic_retriever.invoke(question)
        sources = self._extract_sources(docs)

        return {
            "answer": answer,
            "sources": sources,
            "mode": "basic",
        }

    def query_basic_stream(self, question: str):
        """
        Stream basic vector search query token by token.

        Yields partial answer strings, then yields the final dict with sources.
        """
        if not self.manager.exists:
            raise ValueError("Vector store not found. Run: python -m scripts.ingest")

        self._ensure_basic_chain()

        # Stream tokens
        answer = ""
        for chunk in self._basic_chain.stream(question):
            answer += chunk
            yield {"partial": answer, "mode": "basic"}

        # Append sources at the end
        docs = self._basic_retriever.invoke(question)
        sources = self._extract_sources(docs)
        yield {"answer": answer, "sources": sources, "mode": "basic"}

    def _ensure_hybrid_chain(self):
        """Build hybrid chain if not cached."""
        if not self.manager.exists:
            raise ValueError("Vector store not found. Run: python -m scripts.ingest")
        if self.bm25_index is None:
            raise ValueError("BM25 index not found. Run: python -m scripts.ingest --build-bm25")

        if self._hybrid_chain is None:
            retriever = HybridRetriever(
                vector_retriever=self.manager.get_retriever(k=RETRIEVER_K * 2),
                bm25_index=self.bm25_index,
                k=RETRIEVER_K,
                reranker=self.reranker,
            )
            prompt = get_prompt()
            llm = get_llm()

            self._hybrid_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            self._hybrid_retriever = retriever

    def query_hybrid(self, question: str) -> dict:
        """
        Execute hybrid (vector + BM25) search query.

        Args:
            question: User question

        Returns:
            dict with:
                - answer: str
                - sources: list[dict]
                - mode: "hybrid"
        """
        self._ensure_hybrid_chain()

        # Execute query
        answer = self._hybrid_chain.invoke(question)

        # Get source documents
        docs = self._hybrid_retriever.invoke(question)
        sources = self._extract_sources(docs)

        return {
            "answer": answer,
            "sources": sources,
            "mode": "hybrid",
        }

    def query_hybrid_stream(self, question: str):
        """
        Stream hybrid search query token by token.

        Yields partial answer strings, then yields the final dict with sources.
        """
        self._ensure_hybrid_chain()

        # Stream tokens
        answer = ""
        for chunk in self._hybrid_chain.stream(question):
            answer += chunk
            yield {"partial": answer, "mode": "hybrid"}

        # Append sources at the end
        docs = self._hybrid_retriever.invoke(question)
        sources = self._extract_sources(docs)
        yield {"answer": answer, "sources": sources, "mode": "hybrid"}

    def query_agentic(self, question: str) -> dict:
        """
        Execute agentic (multi-step reasoning) query.

        Args:
            question: User question

        Returns:
            dict with:
                - answer: str
                - sources: list[dict]
                - reasoning_steps: list[str]
                - mode: "agentic"
        """
        if not self.manager.exists:
            raise ValueError("Vector store not found. Run: python -m scripts.ingest")

        # Build agent if not cached
        if self._agent is None:
            self._agent = build_agent(
                self.manager,
                bm25_index=self.bm25_index,
                reranker=self.reranker
            )

        # Execute query
        result = self._agent.invoke({"input": question})

        # Extract reasoning steps
        reasoning_steps = []
        if "intermediate_steps" in result:
            for action, observation in result["intermediate_steps"]:
                tool_name = action.tool if hasattr(action, "tool") else "unknown"
                tool_input = action.tool_input if hasattr(action, "tool_input") else ""

                reasoning_steps.append(f"🔧 Tool: {tool_name}")
                reasoning_steps.append(f"📥 Input: {tool_input}")

                # Truncate long observations
                obs_preview = str(observation)[:300]
                if len(str(observation)) > 300:
                    obs_preview += "..."
                reasoning_steps.append(f"📤 Output: {obs_preview}")
                reasoning_steps.append("")  # blank line

        # Extract sources from agent output (if available)
        sources = []
        # Note: Agent output may not have structured sources,
        # so we provide an empty list for now

        return {
            "answer": result["output"],
            "sources": sources,
            "reasoning_steps": reasoning_steps,
            "mode": "agentic",
        }
