"""CLI: Ingest all PDFs from data/ into ChromaDB.

Usage:
    python -m ingest_test.ingest
    python -m ingest_test.ingest --reset        # wipe store first
    python -m ingest_test.ingest data/single.pdf # one file only
"""
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.progress import track

from . import config
from .converter import convert_pdf
from .chunker import chunk_document
from .metadata import enrich_metadata
from .store import upsert_chunks, reset_store

console = Console()


def compute_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Compute embeddings via your existing Ollama setup.
    Replace this with your actual embedding function.
    """
    import requests
    embeddings = []
    for text in texts:
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        embeddings.append(resp.json()["embedding"])
    return embeddings


def ingest_file(pdf_path: Path) -> int:
    """Process a single PDF through the full pipeline. Returns chunk count."""
    console.print(f"  [cyan]Converting:[/cyan] {pdf_path.name}")
    doc = convert_pdf(pdf_path)

    console.print(f"  [cyan]Chunking:[/cyan]  {pdf_path.name}")
    records = chunk_document(doc, pdf_path.name)

    # Enrich metadata
    for r in records:
        r.metadata = enrich_metadata(r.text, r.metadata)

    # Embed the contextualized text (includes section headers)
    console.print(f"  [cyan]Embedding:[/cyan] {len(records)} chunks")
    embeddings = compute_embeddings([r.contextualized_text for r in records])

    # Store
    upsert_chunks(records, embeddings)
    return len(records)


def main():
    args = sys.argv[1:]
    do_reset = "--reset" in args
    file_args = [a for a in args if not a.startswith("--")]

    if do_reset:
        console.print("[yellow]Resetting vector store...[/yellow]")
        reset_store()

    # Determine files to process
    if file_args:
        pdf_files = [Path(f) for f in file_args]
    else:
        pdf_files = sorted(config.DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[red]No PDFs found in {config.DATA_DIR}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Ingesting {len(pdf_files)} PDF(s)[/bold]\n")
    total_chunks = 0
    t0 = time.time()

    for pdf_path in pdf_files:
        n = ingest_file(pdf_path)
        total_chunks += n
        console.print(f"  [green]✓[/green] {pdf_path.name}: {n} chunks\n")

    elapsed = time.time() - t0
    console.print(f"[bold green]Done:[/bold green] {total_chunks} chunks from "
                  f"{len(pdf_files)} files in {elapsed:.1f}s")


if __name__ == "__main__":
    main()