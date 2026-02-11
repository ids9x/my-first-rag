# Experimental Docling Ingestion

⚠️ **IMPORTANT**: This module requires a **separate virtual environment** due to incompatible dependencies.

## Why Separate?

- **Main system**: Uses `langchain-chroma` (high-level wrapper)
- **This module**: Uses `chromadb` directly (low-level API)
- **Conflict**: Different chromadb versions cannot coexist

## Setup

```bash
# Create isolated environment
python3 -m venv ingest_test/.venv
source ingest_test/.venv/bin/activate
pip install -r ingest_test/requirements.txt

# Run ingestion test
cd ingest_test
python ingest.py
```

## Status

Experimental. Not integrated with main RAG system.

## What is Docling?

Docling is an advanced PDF processing library that provides better document structure extraction compared to PyPDF. This experimental module explores using Docling for improved document chunking and ingestion.

---

## Original Content

(The original README content has been preserved below for reference)

# Docling-Based Ingestion Test

Alternative ingestion pipeline using Docling's advanced PDF processing.

## Features
- Better document structure preservation
- Improved table extraction
- Enhanced multi-column layout handling

## Usage
See instructions above for proper setup with isolated environment.
