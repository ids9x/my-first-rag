"""
Chunking Strategies (Priority 2)

Two strategies:
  1. recursive  — default RecursiveCharacterTextSplitter (good for general docs)
  2. section    — section-aware splitting tuned for regulatory standards
                  (NQA-1, ASME, IAEA, KTA, etc.)

The section strategy uses separators that match common heading patterns
in nuclear regulatory documents, in both English and German.
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SECTION_CHUNK_SIZE,
    SECTION_CHUNK_OVERLAP,
    SECTION_SEPARATORS,
)


def get_splitter(strategy: str = "recursive") -> RecursiveCharacterTextSplitter:
    """
    Return a text splitter based on the chosen strategy.

    Args:
        strategy: "recursive" for general docs, "section" for structured standards.
    """
    if strategy == "section":
        return RecursiveCharacterTextSplitter(
            chunk_size=SECTION_CHUNK_SIZE,
            chunk_overlap=SECTION_CHUNK_OVERLAP,
            separators=SECTION_SEPARATORS,
            keep_separator=True,
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Load a PDF and return raw page documents."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"📄 Loading: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"   {len(pages)} pages loaded.")
    return pages


def load_and_chunk(
    pdf_path: str | Path,
    strategy: str = "recursive",
) -> list[Document]:
    """
    Load a PDF and split it into chunks.

    Args:
        pdf_path: Path to the PDF file.
        strategy: "recursive" or "section".

    Returns:
        List of Document chunks with metadata.
    """
    pages = load_pdf(pdf_path)
    splitter = get_splitter(strategy)
    chunks = splitter.split_documents(pages)

    # Enrich metadata with source filename and strategy used
    for chunk in chunks:
        chunk.metadata["source_file"] = Path(pdf_path).name
        chunk.metadata["chunk_strategy"] = strategy

    print(f"   Split into {len(chunks)} chunks (strategy: {strategy}).")
    return chunks


def load_directory(
    directory: str | Path,
    strategy: str = "recursive",
    skip_existing: set[str] | None = None,
) -> list[Document]:
    """
    Load all PDFs in a directory.

    Args:
        directory: Path to folder containing PDFs.
        strategy: Chunking strategy to apply.
        skip_existing: Set of source filenames to skip (already ingested).

    Returns:
        Combined list of chunks from all PDFs.
    """
    directory = Path(directory)
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  No PDFs found in {directory}")
        return []

    all_chunks = []
    for pdf in pdf_files:
        if skip_existing and pdf.name in skip_existing:
            print(f"⏭️  Skipping {pdf.name} (already ingested)")
            continue
        chunks = load_and_chunk(pdf, strategy=strategy)
        all_chunks.extend(chunks)

    print(f"\n📊 Total: {len(all_chunks)} chunks from {len(pdf_files)} PDFs.")
    return all_chunks
