"""
Parallel + Merge module — runs multiple retrieval strategies concurrently,
deduplicates results, and produces a single merged answer.

This gives the broadest possible coverage:
  - Semantic similarity catches conceptual matches
  - BM25 catches exact keyword matches
  - Reranking catches high-precision matches

Pipeline:
  Query → [Vector, Hybrid, Reranked] in parallel → Merge + Dedup → LLM → Answer
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from rich.console import Console

from core.llm import get_llm
from modules.multilingual import get_prompt
from config.settings import (
    PARALLEL_PER_STRATEGY_K,
    PARALLEL_FINAL_K,
    PARALLEL_MAX_WORKERS,
)

console = Console()


def run_strategy(
    name: str,
    retriever,
    question: str,
) -> list[Document]:
    """Run a single retrieval strategy and return its results.

    This is the unit of work submitted to the thread pool.
    Each strategy is a LangChain retriever with an .invoke() method.

    Args:
        name: Human-readable strategy name (for logging).
        retriever: A LangChain-compatible retriever.
        question: The user's question.

    Returns:
        List of Documents retrieved by this strategy.
    """
    console.print(f"[cyan]Parallel: running '{name}' strategy...[/cyan]")
    docs = retriever.invoke(question)
    console.print(
        f"[cyan]Parallel: '{name}' returned {len(docs)} chunks[/cyan]"
    )
    return docs


def merge_and_deduplicate(
    results: dict[str, list[Document]],
    final_k: int = PARALLEL_FINAL_K,
) -> tuple[list[Document], dict[str, int]]:
    """Merge results from multiple strategies, deduplicate, and rank.

    Documents are deduplicated by content hash. Documents appearing in
    multiple strategy results get a frequency boost to their score.

    Scoring:
      - position_score = 1.0 - (rank / num_results_in_strategy)
      - For each strategy that returns a document, scores are summed.
      - frequency_bonus = 0.1 * (num_strategies_returning_doc - 1)
      - combined_score = sum_of_position_scores + frequency_bonus

    Args:
        results: Dict mapping strategy name → list of Documents.
        final_k: How many documents to keep after merging.

    Returns:
        Tuple of:
            - Deduplicated, ranked list of Documents (top final_k).
            - Dict mapping strategy name → count of unique contributions.
    """
    # Track scores, frequency, and which strategies contributed each doc
    doc_scores: dict[int, float] = {}       # content_hash → combined score
    doc_frequency: dict[int, int] = {}      # content_hash → num strategies
    doc_map: dict[int, Document] = {}       # content_hash → Document
    doc_strategies: dict[int, list[str]] = {}  # content_hash → strategy names

    for strategy_name, docs in results.items():
        num_docs = len(docs) if docs else 1  # avoid division by zero
        for rank, doc in enumerate(docs):
            content_hash = hash(doc.page_content)
            doc_map[content_hash] = doc

            # Position-based score: first result = ~1.0, last = ~0.0
            position_score = 1.0 - (rank / num_docs)
            doc_scores[content_hash] = (
                doc_scores.get(content_hash, 0.0) + position_score
            )

            # Track frequency and contributing strategies
            if content_hash not in doc_frequency:
                doc_frequency[content_hash] = 0
                doc_strategies[content_hash] = []
            doc_frequency[content_hash] += 1
            doc_strategies[content_hash].append(strategy_name)

    # Apply frequency bonus: docs found by multiple strategies rank higher
    for content_hash in doc_scores:
        frequency_bonus = 0.1 * (doc_frequency[content_hash] - 1)
        doc_scores[content_hash] += frequency_bonus

    # Sort by combined score (highest first) and take top final_k
    sorted_hashes = sorted(
        doc_scores, key=lambda h: doc_scores[h], reverse=True
    )
    merged_docs = [doc_map[h] for h in sorted_hashes[:final_k]]

    # Build strategy contribution counts (unique chunks per strategy)
    # A chunk "belongs to" the first strategy that returned it, for counting
    strategy_counts: dict[str, int] = {name: 0 for name in results}
    seen_in_final = set()
    for h in sorted_hashes[:final_k]:
        if h not in seen_in_final:
            seen_in_final.add(h)
            # Credit the first strategy that returned this doc
            for strategy_name in doc_strategies[h]:
                strategy_counts[strategy_name] += 1
                break  # only credit one strategy per doc

    return merged_docs, strategy_counts


def parallel_merge_query(question: str, query_service, chat_history=None) -> dict:
    """Full parallel + merge pipeline.

    Runs multiple retrieval strategies concurrently, merges their results,
    then sends the merged chunk set to the LLM for a final answer.

    This is the main entry point called by QueryService.query_parallel().
    It receives the QueryService instance to access its retrievers and components.

    Args:
        question: The user's natural-language question.
        query_service: A QueryService instance with manager, bm25_index,
                       and reranker attributes.
        chat_history: Optional LangChain message list for multi-turn context.

    Returns:
        dict with:
            - answer: str
            - sources: list[dict]
            - mode: "parallel"
            - strategy_counts: dict (strategy name → chunk count)
            - total_unique_chunks: int
    """
    from modules.reranking import MMRRerankingRetriever
    from modules.hybrid_search import HybridRetriever
    from config.settings import MMR_FETCH_K, MMR_LAMBDA_MULT

    k = PARALLEL_PER_STRATEGY_K

    # Build the strategies dict: name → retriever
    # Only include strategies whose dependencies are available.
    strategies: dict[str, object] = {}

    # Strategy 1: Basic vector similarity (always available)
    strategies["vector"] = query_service.manager.get_retriever(k=k)

    # Strategy 2: Hybrid vector + BM25 (only if BM25 index exists)
    if query_service.bm25_index is not None:
        strategies["hybrid"] = HybridRetriever(
            vector_retriever=query_service.manager.get_retriever(k=k * 2),
            bm25_index=query_service.bm25_index,
            k=k,
            reranker=None,  # don't rerank inside hybrid — merge handles ranking
        )

    # Strategy 3: MMR reranked (if reranker enabled) or basic MMR from ChromaDB
    if query_service.reranker:
        strategies["reranked"] = MMRRerankingRetriever(
            vector_store=query_service.manager.get_store(),
            reranker=query_service.reranker,
            fetch_k=MMR_FETCH_K,
            final_k=k,
            lambda_mult=MMR_LAMBDA_MULT,
        )
    else:
        # Fallback: use ChromaDB's built-in MMR search
        strategies["mmr"] = query_service.manager.get_retriever(
            k=k, search_type="mmr"
        )

    console.print(
        f"[cyan]Parallel: running {len(strategies)} strategies: "
        f"{list(strategies.keys())}[/cyan]"
    )

    # Run all strategies concurrently
    results: dict[str, list[Document]] = {}

    with ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as executor:
        future_to_name = {
            executor.submit(run_strategy, name, retriever, question): name
            for name, retriever in strategies.items()
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result(timeout=120)
            except Exception as e:
                console.print(
                    f"[yellow]Parallel: strategy '{name}' failed: {e}[/yellow]"
                )
                # Skip this strategy — continue with the others

    if not results:
        return {
            "answer": "All retrieval strategies failed. No documents retrieved.",
            "sources": [],
            "mode": "parallel",
            "strategy_counts": {},
            "total_unique_chunks": 0,
        }

    # Merge and deduplicate
    merged_docs, strategy_counts = merge_and_deduplicate(results)
    total_unique = len(merged_docs)

    console.print(
        f"[cyan]Parallel: merged to {total_unique} unique chunks "
        f"from {sum(len(v) for v in results.values())} total[/cyan]"
    )

    if not merged_docs:
        return {
            "answer": "No documents were retrieved by any strategy.",
            "sources": [],
            "mode": "parallel",
            "strategy_counts": strategy_counts,
            "total_unique_chunks": 0,
        }

    # Build context and send to LLM (same prompt as basic/hybrid modes)
    context = "\n\n---\n\n".join(doc.page_content for doc in merged_docs)
    prompt = get_prompt()
    llm = get_llm()

    # Format the prompt with context, question, and optional chat history
    formatted = prompt.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history or [],
    })

    try:
        response = llm.invoke(formatted)
        answer = response.content
    except Exception as e:
        console.print(f"[red]Parallel: LLM call failed: {e}[/red]")
        answer = f"Retrieved {total_unique} chunks but LLM generation failed: {e}"

    # Build sources list
    sources = []
    for doc in merged_docs:
        sources.append({
            "source": doc.metadata.get(
                "source_file", doc.metadata.get("source", "unknown")
            ),
            "page": doc.metadata.get("page", "?"),
        })

    return {
        "answer": answer,
        "sources": sources,
        "mode": "parallel",
        "strategy_counts": strategy_counts,
        "total_unique_chunks": total_unique,
    }
