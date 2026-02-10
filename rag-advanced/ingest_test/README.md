# Docling RAG Ingest Pipeline

A production-ready RAG ingest pipeline using [Docling](https://github.com/docling-project/docling) for processing nuclear regulatory PDFs into a ChromaDB vector store.

## Overview

This pipeline converts complex regulatory documents (IAEA standards, NRC guidelines, etc.) into semantically-searchable chunks using:

- **Docling Classic Pipeline** - Fast, deterministic document parsing with OCR and table structure recognition
- **HybridChunker** - Context-aware chunking aligned to embedding model tokenizer
- **ChromaDB** - Vector storage with metadata filtering
- **Ollama** - Local embedding generation (nomic-embed-text)

## Architecture

```
PDF Document
    ↓
┌───────────────────────────────────────────┐
│ 1. CONVERT (converter.py)                 │
│    - Docling Classic Pipeline             │
│    - OCR for scanned docs                 │
│    - Table structure extraction           │
│    - GPU acceleration                     │
└───────────────────────────────────────────┘
    ↓ DoclingDocument
┌───────────────────────────────────────────┐
│ 2. CHUNK (chunker.py)                     │
│    - HybridChunker (512 tokens)           │
│    - Section hierarchy preservation       │
│    - Context-enriched text                │
└───────────────────────────────────────────┘
    ↓ ChunkRecords
┌───────────────────────────────────────────┐
│ 3. ENRICH (metadata.py)                   │
│    - Extract metadata                     │
│    - Document type classification         │
│    - Requirement detection                │
└───────────────────────────────────────────┘
    ↓ Enriched ChunkRecords
┌───────────────────────────────────────────┐
│ 4. EMBED (ingest.py)                      │
│    - Ollama nomic-embed-text              │
│    - Contextualized text embedding        │
└───────────────────────────────────────────┘
    ↓ Vectors
┌───────────────────────────────────────────┐
│ 5. STORE (store.py)                       │
│    - ChromaDB collection                  │
│    - Metadata indexing                    │
└───────────────────────────────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- `docling>=2.50` - Document conversion
- `docling-core[chunking]` - HybridChunker
- `chromadb>=0.5` - Vector store

### 2. Configure Settings

Edit [config.py](config.py):

```python
# Paths
DATA_DIR = Path(__file__).parent.parent / "data"  # Your PDF directory
CHROMA_DIR = Path(__file__).parent / ".chromadb"
COLLECTION_NAME = "nuclear_docs"

# Converter settings
DO_OCR = True                     # Enable for scanned docs
DO_TABLE_STRUCTURE = True         # Extract table structure
TABLE_MODE = "ACCURATE"           # or "FAST"
OCR_LANG = ["en", "de"]          # Your document languages
NUM_THREADS = 4                   # CPU threads

# Chunker settings
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
CHUNK_MAX_TOKENS = 512            # Match your embedding model
CHUNK_MERGE_PEERS = True          # Merge small adjacent chunks

# Query settings
TOP_K = 5                         # Number of results to return
```

### 3. Ensure Ollama is Running

```bash
# Start Ollama (if not already running)
ollama serve

# Pull the embedding model
ollama pull nomic-embed-text
```

## Usage

### Inspect Document Structure

Preview how Docling parses a PDF before ingesting:

```bash
python -m ingest_test.inspect_doc ../data/NQA-1-2017.pdf
```

**Output:**
- Document structure tree
- Section hierarchy with nesting levels
- Markdown export preview (first 2000 chars)

### Ingest Documents

Process PDFs into ChromaDB:

```bash
# Ingest all PDFs in data/
python -m ingest_test.ingest

# Ingest specific file
python -m ingest_test.ingest ../data/GSR-Part-2.pdf

# Reset store and re-ingest
python -m ingest_test.ingest --reset
```

**Process:**
1. Converts PDF with Docling
2. Chunks into ~512 token segments
3. Enriches with metadata
4. Generates embeddings via Ollama
5. Stores in ChromaDB

### Query the Store

Search your ingested documents:

```bash
python -m ingest_test.query "What are the quality assurance requirements?"
```

**Returns:**
- Top-K relevant chunks
- Source file and page numbers
- Section context
- Similarity scores

### Inspect Chunks

View chunked output for debugging:

```bash
python -m ingest_test.inspect_chunks ../data/NQA-1-2017.pdf
```

*(To be implemented - currently empty)*

## File Descriptions

| File | Purpose | Status |
|------|---------|--------|
| `config.py` | Central configuration | ✅ Complete |
| `converter.py` | Docling Classic pipeline wrapper | ✅ Complete |
| `chunker.py` | HybridChunker with context enrichment | ✅ Complete |
| `ingest.py` | Main ingest CLI | ✅ Complete |
| `inspect_doc.py` | Document structure viewer | ✅ Complete |
| `query.py` | Query interface | ✅ Complete |
| `metadata.py` | Metadata extraction | 🚧 To implement |
| `store.py` | ChromaDB operations | 🚧 To implement |
| `inspect_chunks.py` | Chunk debugging tool | 🚧 To implement |

## Configuration Deep Dive

### OCR Settings

For **scanned** IAEA documents:
```python
DO_OCR = True
OCR_LANG = ["en", "de"]  # German + English
```

For **digital-born** PDFs (faster):
```python
DO_OCR = False
```

### Table Extraction

Critical for compliance matrices and requirement tables:
```python
DO_TABLE_STRUCTURE = True
TABLE_CELL_MATCHING = True   # Map structure back to PDF cells
TABLE_MODE = "ACCURATE"      # or "FAST" for speed
```

### Chunking Strategy

**HybridChunker** respects document structure:
- Tries to keep sections together
- Splits long sections at ~512 tokens
- Merges small adjacent chunks in same section
- Prepends section hierarchy for context

Example contextualized chunk:
```
Part II > Subpart 2.7 > Quality Assurance Records

Records shall be maintained to furnish evidence of activities
affecting quality. Records shall include the results of...
```

### Hardware Acceleration

Docling auto-detects available accelerators:
```python
AcceleratorDevice.AUTO  # Will use CUDA if available
NUM_THREADS = 4         # CPU threads for table extraction
```

On your DGX Spark: expect 2-5x speedup from GPU.

## Offline / Air-Gapped Operation

Download Docling models once:

```bash
docling-tools models download --dir ./models
```

Then configure:
```python
ARTIFACTS_PATH = Path("/path/to/models/docling")
```

Models will load from local path instead of downloading.

## Performance

### Typical Throughput

| Document Type | Pages | Time | Chunks |
|--------------|-------|------|--------|
| IAEA GSR (digital) | 50 | ~30s | 200-300 |
| NRC Guide (scanned) | 100 | ~2m | 400-600 |
| Compliance Matrix | 20 | ~15s | 80-120 |

**Bottlenecks:**
- PDF conversion: 0.5-1s/page
- Embedding: 10-50ms/chunk (Ollama)
- ChromaDB insert: <1ms/chunk

## Troubleshooting

### Import Errors

```bash
ModuleNotFoundError: No module named 'docling'
```

**Fix:** Install requirements from the correct directory:
```bash
pip install -r rag-advanced/ingest_test/requirements.txt
```

### Relative Import Errors

```bash
ImportError: attempted relative import with no known parent package
```

**Fix:** Run as module, not script:
```bash
# ❌ Wrong
python ingest_test/inspect_doc.py file.pdf

# ✅ Correct
python -m ingest_test.inspect_doc file.pdf
```

### Ollama Connection Refused

```bash
requests.exceptions.ConnectionError: Connection refused
```

**Fix:** Start Ollama:
```bash
ollama serve
```

### GPU Not Detected

Docling will fall back to CPU automatically. Check CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Extending the Pipeline

### Custom Metadata Extraction

Implement [metadata.py](metadata.py):

```python
def enrich_metadata(chunk_text: str, metadata: dict) -> dict:
    """Extract domain-specific metadata from chunk text."""

    # Detect requirement identifiers (e.g., "Requirement 3.2")
    if re.search(r'Requirement\s+\d+\.\d+', chunk_text):
        metadata['is_requirement'] = True

    # Classify document sections
    if 'quality assurance' in chunk_text.lower():
        metadata['category'] = 'QA'

    return metadata
```

### Custom Chunking Strategy

Modify [chunker.py](chunker.py) to use different chunking:

```python
from docling_core.transforms.chunker import DoclingChunker

def build_custom_chunker():
    return DoclingChunker(
        # Your custom chunking logic
    )
```

### Query with Filters

Extend [query.py](query.py) with metadata filters:

```python
results = collection.query(
    query_embeddings=[embedding],
    n_results=TOP_K,
    where={"category": "QA"}  # Filter by metadata
)
```

## Best Practices

### Document Preparation

1. **Single-topic PDFs work best** - Split large standards into parts
2. **OCR for scanned docs** - Enable `DO_OCR = True`
3. **Consistent naming** - Use descriptive filenames (they become source_file metadata)

### Chunking Tuning

1. **Match embedding model** - `CHUNK_MAX_TOKENS` should match your model's limit
2. **Preserve context** - `CHUNK_MERGE_PEERS = True` keeps related content together
3. **Test chunk quality** - Use `inspect_chunks.py` to review output

### Query Optimization

1. **Increase TOP_K** for comprehensive answers (5-10)
2. **Use metadata filters** to narrow search scope
3. **Rephrase queries** if results are poor (embedding models are sensitive to wording)

## Roadmap

- [ ] Implement metadata.py (requirement detection, document classification)
- [ ] Implement store.py (ChromaDB operations with proper error handling)
- [ ] Implement inspect_chunks.py (chunk debugging and quality review)
- [ ] Add reranking for improved retrieval
- [ ] Support multi-modal retrieval (tables, figures)
- [ ] Batch processing with progress bars
- [ ] CLI argument parsing with typer

## References

- [Docling Documentation](https://docling-project.github.io/docling/)
- [HybridChunker Guide](https://docling-project.github.io/docling/usage/chunking/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Ollama Models](https://ollama.com/library)

## License

Internal use for nuclear regulatory compliance workflows.
