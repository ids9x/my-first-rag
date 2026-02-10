"""CLI: Convert + chunk a PDF, print chunks with metadata.

Usage:
    python -m ingest_test.inspect_chunks data/NQA-1-2017.pdf
    python -m ingest_test.inspect_chunks data/NQA-1-2017.pdf --max 10
"""
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .converter import convert_pdf
from .chunker import chunk_document
from .metadata import enrich_metadata

console = Console()


def main():
    max_display = 999
    pdf_path = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--max" and i + 1 < len(args):
            max_display = int(args[i + 1])
        elif not arg.startswith("--"):
            pdf_path = Path(arg)

    if not pdf_path or not pdf_path.exists():
        console.print("[red]Usage: python -m ingest_test.inspect_chunks <pdf_path> [--max N][/red]")
        sys.exit(1)

    console.print(f"\n[bold]Converting:[/bold] {pdf_path.name}")
    doc = convert_pdf(pdf_path)

    console.print(f"[bold]Chunking...[/bold]")
    records = chunk_document(doc, pdf_path.name)

    # Enrich with nuclear metadata
    for r in records:
        r.metadata = enrich_metadata(r.text, r.metadata)

    console.print(f"\n[bold green]Total chunks: {len(records)}[/bold green]\n")

    for r in records[:max_display]:
        # Summary table
        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column(style="bold cyan", width=20)
        tbl.add_column()
        tbl.add_row("ID", r.id)
        tbl.add_row("Section", r.metadata.get("section_path", "—"))
        tbl.add_row("Page", str(r.metadata.get("page", "—")))
        tbl.add_row("Doc type", r.metadata.get("doc_type", "—"))
        tbl.add_row("Standards", r.metadata.get("standards_referenced", "—"))
        tbl.add_row("Cross-refs", r.metadata.get("cross_references", "—"))

        preview = r.text[:300].replace("\n", " ")
        console.print(Panel(tbl, title=f"Chunk {r.metadata['chunk_index']}", width=100))
        console.print(f"  [dim]{preview}{'...' if len(r.text) > 300 else ''}[/dim]\n")


if __name__ == "__main__":
    main()