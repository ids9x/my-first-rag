# Advanced RAG Pipeline for DGX Spark

A modular, production-ready RAG system built on top of your first RAG guide.
Designed for nuclear regulatory documents (NQA-1, ASME, IAEA standards) in German and English.

## Architecture

```
rag-advanced/
├── config/
│   ├── __init__.py
│   └── settings.py          # All configuration in one place
├── core/
│   ├── __init__.py
│   ├── embeddings.py         # Embedding model management
│   ├── store.py              # Persistent ChromaDB vector store
│   └── llm.py                # LLM wrapper
├── modules/
│   ├── __init__.py
│   ├── chunking.py           # P2: Section-aware + recursive chunking
│   ├── hybrid_search.py      # P3: BM25 + vector similarity fusion
│   ├── multilingual.py       # P4: DE/EN embedding & query routing
│   ├── agentic.py            # P5: Multi-step reasoning agents
│   └── knowledge_graph.py    # P6: txt2kg relationship extraction
├── scripts/
│   ├── ingest.py             # Ingest PDFs into the vector store
│   ├── query.py              # Interactive query loop
│   └── reset_store.py        # Wipe and rebuild the vector store
├── data/                     # Place your PDFs here
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Clone/copy to your Spark
scp -r rag-advanced/ username@spark.local:~/

# 2. Create venv and install
cd ~/rag-advanced
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Place PDFs in data/
cp /path/to/your/*.pdf data/

# 4. Ingest documents
python -m scripts.ingest

# 5. Query interactively
python -m scripts.query
```

## Priority Roadmap

| # | Module | Status | Description |
|---|--------|--------|-------------|
| 1 | Persistent Store | ✅ Ready | Skip re-embedding on subsequent runs |
| 2 | Chunking Strategies | ✅ Ready | Section-aware splitting for standards |
| 3 | Hybrid Search | ✅ Ready | BM25 + vector fusion retrieval |
| 4 | Multilingual | ✅ Ready | DE/EN document handling |
| 5 | Agentic RAG | ✅ Ready | Multi-step reasoning with tools |
| 6 | Knowledge Graph | ✅ Ready | Relationship extraction (txt2kg) |

## Testing with Small Models

Everything defaults to lightweight models for testing:
- **Chat**: `gemma3:4b` (~2.5 GB VRAM)
- **Embeddings**: `nomic-embed-text` (~274 MB)
- **Data**: Works with 1-5 small PDFs

When ready to scale, just change `CHAT_MODEL` in `config/settings.py`.

## Module Usage

Each module can be used independently:

```python
# Hybrid search
python -m scripts.query --mode hybrid

# Multilingual query (auto-detects language)
python -m scripts.query --mode multilingual

# Agentic mode (multi-step reasoning)
python -m scripts.query --mode agentic
```
