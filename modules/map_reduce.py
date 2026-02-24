"""
Map-Reduce module — processes each retrieved chunk independently, then synthesises.

Instead of stuffing all retrieved chunks into a single prompt (which can lose
detail with many chunks), map-reduce processes each chunk independently before
synthesising. This forces the LLM to engage with every chunk individually,
producing more thorough answers for dense nuclear regulatory documents.

Pipeline:
  Query → Retrieve N chunks → Map (parallel LLM calls) → Reduce (synthesise)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console

from core.llm import get_llm
from config.settings import (
    MAP_REDUCE_FETCH_K,
    MAP_REDUCE_MAX_WORKERS,
    MAP_REDUCE_TEMPERATURE,
)

console = Console()

# ---------------------------------------------------------------------------
# Prompt templates (plain strings — formatted with .format() before LLM call)
# ---------------------------------------------------------------------------

MAP_PROMPT = """\
You are analysing a nuclear regulatory document excerpt.

Question: {question}

Document excerpt (from {source_file}, page {page}):
{chunk_text}

Extract ONLY the information from this excerpt that is relevant to the question.
If this excerpt contains no relevant information, respond with "No relevant information."
Be precise and preserve technical terminology."""

REDUCE_PROMPT = """\
You are synthesising information from multiple nuclear regulatory document excerpts.

Question: {question}

Extracted information from {n} document excerpts:

{numbered_summaries}

Synthesise these into a comprehensive answer. Cite sources by their excerpt number.
If excerpts contain contradictory information, note the contradiction.
If no excerpts contained relevant information, say so clearly."""


def map_single_chunk(question: str, doc, llm) -> str:
    """Extract relevant information from a single document chunk.

    Sends one chunk to the LLM with the map prompt. The LLM is asked to
    extract only the information relevant to the user's question.

    Args:
        question: The user's natural-language question.
        doc: A LangChain Document with .page_content and .metadata.
        llm: A configured LLM instance.

    Returns:
        The LLM's extraction (a string summary of relevant info from this chunk).
    """
    source_file = doc.metadata.get(
        "source_file", doc.metadata.get("source", "unknown")
    )
    page = doc.metadata.get("page", "?")

    prompt_text = MAP_PROMPT.format(
        question=question,
        source_file=source_file,
        page=page,
        chunk_text=doc.page_content,
    )

    response = llm.invoke(prompt_text)
    return response.content


def map_all_chunks(
    question: str,
    docs: list,
    llm,
    max_workers: int = MAP_REDUCE_MAX_WORKERS,
) -> list[dict]:
    """Run the map phase over all chunks in parallel.

    Each chunk is sent to the LLM independently via a thread pool.
    If a single map call fails, it is skipped with a warning (the other
    chunks still contribute to the final answer).

    Args:
        question: The user's natural-language question.
        docs: List of LangChain Documents to process.
        llm: A configured LLM instance.
        max_workers: Number of parallel threads for map calls.

    Returns:
        List of dicts, each with:
            - summary: str (the LLM's extraction for that chunk)
            - source_file: str
            - page: str or int
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all map tasks — keyed by future so we can match results to docs
        future_to_doc = {
            executor.submit(map_single_chunk, question, doc, llm): doc
            for doc in docs
        }

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            source_file = doc.metadata.get(
                "source_file", doc.metadata.get("source", "unknown")
            )
            page = doc.metadata.get("page", "?")

            try:
                summary = future.result(timeout=120)
                results.append({
                    "summary": summary,
                    "source_file": source_file,
                    "page": page,
                })
            except Exception as e:
                console.print(
                    f"[yellow]Map-Reduce: map failed for "
                    f"{source_file} p.{page}: {e}[/yellow]"
                )
                # Skip this chunk — continue with the others

    return results


def map_reduce_query(question: str, retriever, llm) -> dict:
    """Full map-reduce pipeline: retrieve, map each chunk, then reduce.

    This is the main entry point called by QueryService.query_map_reduce().
    It receives its dependencies (retriever, llm) rather than constructing
    them, keeping the module testable.

    Args:
        question: The user's natural-language question.
        retriever: A LangChain retriever (vector, hybrid, or reranked).
        llm: A configured LLM instance.

    Returns:
        dict with:
            - answer: str (the synthesised final answer)
            - sources: list[dict] (source_file + page for each chunk)
            - mode: "map_reduce"
            - map_summaries: list[str] (individual chunk summaries)
            - chunk_count: int (how many chunks were processed)
    """
    # Step 1: Retrieve chunks
    console.print(f"[cyan]Map-Reduce: retrieving chunks...[/cyan]")
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "No documents were retrieved for this question.",
            "sources": [],
            "mode": "map_reduce",
            "map_summaries": [],
            "chunk_count": 0,
        }

    console.print(f"[cyan]Map-Reduce: mapping {len(docs)} chunks...[/cyan]")

    # Step 2: Map phase — process each chunk independently (parallel)
    map_results = map_all_chunks(question, docs, llm)

    if not map_results:
        # All map calls failed — fall back to a simple stuffed-context approach
        console.print(
            "[yellow]Map-Reduce: all map calls failed, "
            "falling back to direct context[/yellow]"
        )
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        fallback_prompt = (
            f"Answer this question based on the following context:\n\n"
            f"{context}\n\nQuestion: {question}"
        )
        try:
            fallback_response = llm.invoke(fallback_prompt)
            answer = fallback_response.content
        except Exception as e:
            answer = f"Map-reduce failed and fallback also failed: {e}"

        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get(
                    "source_file", doc.metadata.get("source", "unknown")
                ),
                "page": doc.metadata.get("page", "?"),
            })

        return {
            "answer": answer,
            "sources": sources,
            "mode": "map_reduce",
            "map_summaries": [],
            "chunk_count": len(docs),
        }

    # Step 3: Build numbered summaries for the reduce prompt
    numbered_summaries = ""
    map_summaries_list = []
    for i, mr in enumerate(map_results, 1):
        label = f"[Excerpt {i}: {mr['source_file']}, page {mr['page']}]"
        numbered_summaries += f"{label}\n{mr['summary']}\n\n"
        map_summaries_list.append(f"{label} {mr['summary']}")

    # Step 4: Reduce phase — synthesise all map summaries into a final answer
    console.print(
        f"[cyan]Map-Reduce: reducing {len(map_results)} summaries...[/cyan]"
    )
    reduce_prompt_text = REDUCE_PROMPT.format(
        question=question,
        n=len(map_results),
        numbered_summaries=numbered_summaries,
    )

    try:
        reduce_response = llm.invoke(reduce_prompt_text)
        answer = reduce_response.content
    except Exception as e:
        console.print(f"[red]Map-Reduce: reduce phase failed: {e}[/red]")
        # Return the map summaries as the answer if reduce fails
        answer = (
            "The reduce (synthesis) step failed. "
            "Here are the individual chunk summaries:\n\n"
            + numbered_summaries
        )

    # Step 5: Build sources list from map results
    sources = []
    for mr in map_results:
        sources.append({
            "source": mr["source_file"],
            "page": mr["page"],
        })

    return {
        "answer": answer,
        "sources": sources,
        "mode": "map_reduce",
        "map_summaries": map_summaries_list,
        "chunk_count": len(map_results),
    }
