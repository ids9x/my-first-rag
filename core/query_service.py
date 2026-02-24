"""
Service layer for RAG queries.

Provides clean, reusable functions for Basic, Hybrid, and Agentic query modes.
Used by both CLI (scripts/query.py) and web UI (app.py).
"""
from pathlib import Path
from operator import itemgetter
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
    MAP_REDUCE_FETCH_K,
    MAP_REDUCE_TEMPERATURE,
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
                {
                    "context": itemgetter("question") | retriever | format_docs,
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("chat_history"),
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            self._basic_retriever = retriever

    def query_basic(self, question: str, chat_history: list | None = None) -> dict:
        """
        Execute basic vector search query.

        Args:
            question: User question
            chat_history: Optional LangChain message list for multi-turn context

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
        input_dict = {"question": question, "chat_history": chat_history or []}
        answer = self._basic_chain.invoke(input_dict)

        # Get source documents
        docs = self._basic_retriever.invoke(question)
        sources = self._extract_sources(docs)

        return {
            "answer": answer,
            "sources": sources,
            "mode": "basic",
        }

    def query_basic_stream(self, question: str, chat_history: list | None = None):
        """
        Stream basic vector search query token by token.

        Yields partial answer strings, then yields the final dict with sources.
        """
        if not self.manager.exists:
            raise ValueError("Vector store not found. Run: python -m scripts.ingest")

        self._ensure_basic_chain()

        # Stream tokens
        input_dict = {"question": question, "chat_history": chat_history or []}
        answer = ""
        for chunk in self._basic_chain.stream(input_dict):
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
                {
                    "context": itemgetter("question") | retriever | format_docs,
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("chat_history"),
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            self._hybrid_retriever = retriever

    def query_hybrid(self, question: str, chat_history: list | None = None) -> dict:
        """
        Execute hybrid (vector + BM25) search query.

        Args:
            question: User question
            chat_history: Optional LangChain message list for multi-turn context

        Returns:
            dict with:
                - answer: str
                - sources: list[dict]
                - mode: "hybrid"
        """
        self._ensure_hybrid_chain()

        # Execute query
        input_dict = {"question": question, "chat_history": chat_history or []}
        answer = self._hybrid_chain.invoke(input_dict)

        # Get source documents
        docs = self._hybrid_retriever.invoke(question)
        sources = self._extract_sources(docs)

        return {
            "answer": answer,
            "sources": sources,
            "mode": "hybrid",
        }

    def query_hybrid_stream(self, question: str, chat_history: list | None = None):
        """
        Stream hybrid search query token by token.

        Yields partial answer strings, then yields the final dict with sources.
        """
        self._ensure_hybrid_chain()

        # Stream tokens
        input_dict = {"question": question, "chat_history": chat_history or []}
        answer = ""
        for chunk in self._hybrid_chain.stream(input_dict):
            answer += chunk
            yield {"partial": answer, "mode": "hybrid"}

        # Append sources at the end
        docs = self._hybrid_retriever.invoke(question)
        sources = self._extract_sources(docs)
        yield {"answer": answer, "sources": sources, "mode": "hybrid"}

    def query_agentic(self, question: str, chat_history: list | None = None) -> dict:
        """
        Execute agentic (multi-step reasoning) query.

        Args:
            question: User question
            chat_history: Optional LangChain message list for multi-turn context

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
        result = self._agent.invoke({
            "input": question,
            "chat_history": chat_history or [],
        })

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

    def query_router(self, question: str, chat_history: list | None = None) -> dict:
        """Auto-route to best pipeline via LLM classification.

        Uses a lightweight LLM call to classify the question type, then
        dispatches to Basic, Hybrid, or Agentic mode automatically.

        Args:
            question: User question
            chat_history: Optional LangChain message list for multi-turn context

        Returns:
            dict with:
                - answer: str
                - sources: list[dict]
                - mode: "router"
                - routed_to: str (which pipeline was chosen)
                - classification_reasoning: str (why)
        """
        from modules.router import route_and_execute  # lazy import

        return route_and_execute(question, self, chat_history=chat_history)

    def query_map_reduce(self, question: str) -> dict:
        """Map-reduce: LLM processes each chunk independently, then synthesises.

        Retrieves MAP_REDUCE_FETCH_K chunks (more than standard), runs a
        parallel map phase over each, then reduces into a final answer.

        Args:
            question: User question

        Returns:
            dict with:
                - answer: str
                - sources: list[dict]
                - mode: "map_reduce"
                - map_summaries: list[str]
                - chunk_count: int
        """
        from modules.map_reduce import map_reduce_query  # lazy import

        if not self.manager.exists:
            raise ValueError("Vector store not found. Run: python -m scripts.ingest")

        # Build a retriever that fetches more chunks than standard (8 vs 4).
        # Use the best available strategy: reranked > hybrid > basic vector.
        if self.reranker:
            retriever = MMRRerankingRetriever(
                vector_store=self.manager.get_store(),
                reranker=self.reranker,
                fetch_k=MMR_FETCH_K,
                final_k=MAP_REDUCE_FETCH_K,
                lambda_mult=MMR_LAMBDA_MULT,
            )
        elif self.bm25_index is not None:
            retriever = HybridRetriever(
                vector_retriever=self.manager.get_retriever(
                    k=MAP_REDUCE_FETCH_K * 2
                ),
                bm25_index=self.bm25_index,
                k=MAP_REDUCE_FETCH_K,
                reranker=None,
            )
        else:
            retriever = self.manager.get_retriever(k=MAP_REDUCE_FETCH_K)

        llm = get_llm(temperature=MAP_REDUCE_TEMPERATURE)

        return map_reduce_query(question, retriever, llm)

    def query_parallel(self, question: str, chat_history: list | None = None) -> dict:
        """Parallel retrieval + merge: broadest coverage from multiple strategies.

        Runs vector, hybrid, and reranked retrieval concurrently, merges
        and deduplicates the results, then sends the merged chunks to the LLM.

        Args:
            question: User question
            chat_history: Optional LangChain message list for multi-turn context

        Returns:
            dict with:
                - answer: str
                - sources: list[dict]
                - mode: "parallel"
                - strategy_counts: dict (strategy → chunk count)
                - total_unique_chunks: int
        """
        from modules.parallel import parallel_merge_query  # lazy import

        if not self.manager.exists:
            raise ValueError("Vector store not found. Run: python -m scripts.ingest")

        return parallel_merge_query(question, self, chat_history=chat_history)
