"""CLI: Interactive query against the Docling-ingested store.

Usage:
    python -m ingest_test.query
    python -m ingest_test.query --filter doc_type=requirement
"""
import sys
from rich.console import Console
from rich.panel import Panel

from . import config
from .store import query_store

console = Console()


def compute_query_embedding(text: str) -> list[float]:
    """Embed a query string. Replace with your actual embedding function."""
    import requests
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
    )
    return resp.json()["embedding"]


def parse_filter(args: list[str]) -> dict | None:
    """Parse --filter key=value into a ChromaDB where clause."""
    for i, arg in enumerate(args):
        if arg == "--filter" and i + 1 < len(args):
            key, val = args[i + 1].split("=", 1)
            return {key: val}
    return None


def main():
    where_filter = parse_filter(sys.argv[1:])
    if where_filter:
        console.print(f"[dim]Active filter: {where_filter}[/dim]")

    console.print("[bold]Docling RAG Query Loop[/bold]")
    console.print("[dim]Type 'quit' to exit[/dim]\n")

    while True:
        query = console.input("[bold cyan]Query>[/bold cyan] ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        embedding = compute_query_embedding(query)
        results = query_store(embedding, where_filter=where_filter)

        if not results["documents"][0]:
            console.print("[yellow]No results found.[/yellow]\n")
            continue

        for j, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            section = meta.get("section_path", "—")
            source = meta.get("source_file", "—")
            doc_type = meta.get("doc_type", "—")
            score = 1 - dist  # cosine distance → similarity

            header = (f"[bold]#{j+1}[/bold] ({score:.3f}) "
                     f"[cyan]{source}[/cyan] → {section}")
            console.print(Panel(
                doc[:500] + ("..." if len(doc) > 500 else ""),
                title=header,
                subtitle=f"type={doc_type}",
                width=110,
            ))
        console.print()


if __name__ == "__main__":
    main()