"""
Chunking Strategies (Priority 2)

Two strategies:
  1. recursive  — default RecursiveCharacterTextSplitter (good for general docs)
  2. section    — section-aware splitting tuned for regulatory standards
                  (NQA-1, ASME, IAEA, KTA, etc.)

The section strategy uses separators that match common heading patterns
in nuclear regulatory documents, in both English and German.

Supports loading multiple document formats (PDF, DOCX, XLSX, Email, TXT)
via the loaders module.
"""
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from modules.loaders import load_document, SUPPORTED_EXTENSIONS
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
    """Load a PDF and return raw page documents (kept for backward compat)."""
    return load_document(Path(pdf_path))


def load_and_chunk(
    file_path: str | Path,
    strategy: str = "recursive",
) -> list[Document]:
    """
    Load a document (any supported format) and split it into chunks.

    Args:
        file_path: Path to the document file (PDF, DOCX, XLSX, EML, MSG, TXT).
        strategy: "recursive" or "section".

    Returns:
        List of Document chunks with metadata.
    """
    pages = load_document(file_path)
    splitter = get_splitter(strategy)
    chunks = splitter.split_documents(pages)

    # Enrich metadata with source filename and strategy used
    for chunk in chunks:
        chunk.metadata["source_file"] = Path(file_path).name
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

    # Collect all supported file types
    all_files = []
    for ext in sorted(SUPPORTED_EXTENSIONS):
        all_files.extend(directory.glob(f"*{ext}"))
    all_files = sorted(set(all_files))

    if not all_files:
        print(f"⚠️  No supported documents found in {directory}")
        return []

    all_chunks = []
    for doc_file in all_files:
        if skip_existing and doc_file.name in skip_existing:
            print(f"⏭️  Skipping {doc_file.name} (already ingested)")
            continue
        chunks = load_and_chunk(doc_file, strategy=strategy)
        all_chunks.extend(chunks)

    print(f"\n📊 Total: {len(all_chunks)} chunks from {len(all_files)} documents.")
    return all_chunks
