"""CLI: Convert a PDF with Docling and print its structure tree.

Usage:
    python -m ingest_test.inspect_doc data/NQA-1-2017.pdf
"""
import sys
from pathlib import Path
from rich.console import Console
from rich.tree import Tree

from .converter import convert_pdf

console = Console()


def build_tree(doc) -> Tree:
    """Walk the DoclingDocument and build a Rich tree for display."""
    tree = Tree(f"[bold]{doc.name or 'Document'}[/bold]")

    for item, level in doc.iterate_items():
        label = item.label if hasattr(item, "label") else type(item).__name__
        text_preview = ""
        if hasattr(item, "text") and item.text:
            text_preview = item.text[:80].replace("\n", " ")
            text_preview = f"  [dim]{text_preview}...[/dim]" if len(item.text) > 80 else f"  [dim]{text_preview}[/dim]"

        node_label = f"[cyan]{label}[/cyan] (L{level}){text_preview}"
        tree.add(node_label)

    return tree


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m ingest_test.inspect_doc <pdf_path>[/red]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        console.print(f"[red]File not found: {pdf_path}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Converting:[/bold] {pdf_path.name}")
    doc = convert_pdf(pdf_path)

    console.print(f"\n[bold]Document structure:[/bold]")
    tree = build_tree(doc)
    console.print(tree)

    # Summary stats
    console.print(f"\n[bold]Export preview (first 2000 chars of Markdown):[/bold]")
    md = doc.export_to_markdown()
    console.print(md[:2000])
    console.print(f"\n[dim]... ({len(md)} total characters)[/dim]")


if __name__ == "__main__":
    main()