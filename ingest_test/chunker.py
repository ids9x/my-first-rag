"""Chunk a DoclingDocument using Docling's HybridChunker."""
from dataclasses import dataclass, field
from docling.chunking import HybridChunker

from . import config


@dataclass
class ChunkRecord:
    """A chunk with its text, context-enriched text, and metadata."""
    id: str                              # unique chunk ID
    text: str                            # raw chunk text
    contextualized_text: str             # text with section headers prepended
    metadata: dict = field(default_factory=dict)


def build_chunker() -> HybridChunker:
    """Create a HybridChunker aligned to the embedding model tokenizer."""
    return HybridChunker(
        tokenizer=config.EMBEDDING_MODEL_ID,
        max_tokens=config.CHUNK_MAX_TOKENS,
        merge_peers=config.CHUNK_MERGE_PEERS,
    )


def chunk_document(doc, source_filename: str) -> list[ChunkRecord]:
    """
    Chunk a DoclingDocument and return enriched ChunkRecords.

    The contextualized_text includes section headers prepended,
    which is what you should embed. The raw text is stored for display.
    """
    chunker = build_chunker()
    records = []

    for i, chunk in enumerate(chunker.chunk(doc)):
        # contextualize() prepends the section hierarchy to the chunk text
        # e.g. "Part II > Subpart 2.7 > Quality Assurance Records\n\n<chunk text>"
        ctx_text = chunker.contextualize(chunk)

        # Extract metadata from chunk.meta
        meta = {
            "source_file": source_filename,
            "chunk_index": i,
        }

        # Section path (the hierarchy breadcrumb)
        if hasattr(chunk, "meta") and chunk.meta:
            headings = chunk.meta.headings if hasattr(chunk.meta, "headings") else []
            meta["section_path"] = " > ".join(headings) if headings else ""
            if hasattr(chunk.meta, "page"):
                meta["page"] = chunk.meta.page

        records.append(ChunkRecord(
            id=f"{source_filename}::chunk_{i:04d}",
            text=chunk.text,
            contextualized_text=ctx_text,
            metadata=meta,
        ))

    return records