"""
Agentic RAG (Priority 5)

Adds multi-step reasoning on top of basic retrieval.
The agent can:
  1. Search the vector store (semantic retrieval)
  2. Search with BM25 (keyword retrieval)
  3. Compare information across multiple retrieved passages
  4. Decide if it needs more context before answering

This uses LangChain's tool-calling agent pattern. The LLM decides
which tools to call and when it has enough information to answer.

Note: Requires a model with tool-calling support. gemma3:4b works
for testing but larger models (qwen2.5:32b, nemotron) give much
better agentic behavior.
"""
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from core.store import VectorStoreManager
from core.llm import get_llm
from modules.hybrid_search import BM25Index
from config.settings import MAX_AGENT_STEPS, RETRIEVER_K


def _format_docs(docs: list[Document]) -> str:
    """Format documents into a readable string with source info."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i}: {source}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_agent(
    vector_store_manager: VectorStoreManager,
    bm25_index: BM25Index | None = None,
):
    """
    Build an agentic RAG with retrieval tools.

    Args:
        vector_store_manager: Initialized VectorStoreManager.
        bm25_index: Optional BM25Index for keyword search.

    Returns:
        AgentExecutor ready to invoke.
    """
    retriever = vector_store_manager.get_retriever(k=RETRIEVER_K)

    @tool
    def semantic_search(query: str) -> str:
        """Search the document store using semantic similarity.
        Use this for conceptual questions or paraphrased queries.
        Returns relevant passages from the ingested documents."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return _format_docs(docs)

    @tool
    def keyword_search(query: str) -> str:
        """Search the document store using exact keyword matching.
        Use this when looking for specific terms, section numbers,
        clause references, or exact phrases like 'Section 18' or 'QA Level 1'.
        Returns relevant passages from the ingested documents."""
        if bm25_index is None:
            return "Keyword search is not available (no BM25 index loaded)."
        docs = bm25_index.search(query, k=RETRIEVER_K)
        if not docs:
            return "No documents matched those keywords."
        return _format_docs(docs)

    tools = [semantic_search, keyword_search]

    # Agent prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert assistant for nuclear regulatory documents.
You have access to a document store containing technical standards and regulations.

Your approach:
1. Use semantic_search for conceptual questions.
2. Use keyword_search when the user references specific sections, clauses, or terms.
3. You may call tools multiple times if the first results are insufficient.
4. Always cite which source documents your answer comes from.
5. If you cannot find enough information, say so clearly.

Be precise and factual. These are regulatory documents — accuracy matters.""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = get_llm()

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=MAX_AGENT_STEPS,
        verbose=True,  # Set to False to hide reasoning steps
        handle_parsing_errors=True,
    )
