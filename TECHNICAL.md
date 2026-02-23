# Technical Documentation

This document describes the technical architecture and Python implementation of the Advanced RAG Pipeline — a modular Retrieval-Augmented Generation system built for nuclear regulatory document retrieval.

---

## Table of Contents

- [System Overview](#system-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
  - [Configuration Layer](#configuration-layer)
  - [Embedding Layer](#embedding-layer)
  - [LLM Layer](#llm-layer)
  - [Vector Store](#vector-store)
  - [Query Service](#query-service)
- [Feature Modules](#feature-modules)
  - [Document Chunking](#document-chunking)
  - [Hybrid Search](#hybrid-search)
  - [Cross-Encoder Reranking](#cross-encoder-reranking)
  - [MMR Diversity Selection](#mmr-diversity-selection)
  - [Agentic Reasoning](#agentic-reasoning)
  - [Knowledge Graph](#knowledge-graph)
- [Retrieval Pipelines](#retrieval-pipelines)
  - [Basic Vector Search](#basic-vector-search)
  - [Hybrid Vector + BM25](#hybrid-vector--bm25)
  - [Reranking + MMR Pipeline](#reranking--mmr-pipeline)
  - [Agentic Multi-Step Pipeline](#agentic-multi-step-pipeline)
- [Web Interface](#web-interface)
- [External Services](#external-services)
- [Data Flow](#data-flow)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Query Pipeline](#query-pipeline)
- [Key Design Decisions](#key-design-decisions)

---

## System Overview

The system implements a multi-strategy RAG pipeline that ingests PDF documents, stores them as vector embeddings in ChromaDB, and provides four distinct retrieval modes: basic vector search, hybrid (vector + BM25 keyword) search, hybrid with cross-encoder reranking and MMR diversity selection, and agentic multi-step reasoning.

All components are written in Python 3.12 and orchestrated through LangChain. The system exposes both a CLI and a Gradio web interface. Two external inference servers are required at runtime: Ollama for embeddings and llama-server (llama.cpp) for chat generation.

---

## Technology Stack

### Python Dependencies

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Orchestration** | `langchain` | >=0.3.0 | Chain and agent framework |
| | `langchain-chroma` | >=0.2.0 | ChromaDB vector store integration |
| | `langchain-ollama` | >=0.2.0 | Ollama embeddings integration |
| | `langchain-openai` | >=0.3.0 | OpenAI-compatible API client (for llama-server) |
| | `langchain-text-splitters` | >=0.3.0 | Text chunking strategies |
| | `langchain-community` | >=0.3.0 | Community integrations (PyPDF loader) |
| **Document Loading** | `pypdf` | >=4.0.0 | PDF parsing |
| **Keyword Search** | `rank-bm25` | >=0.2.2 | BM25Okapi keyword index |
| **Reranking** | `FlagEmbedding` | >=1.2.0 | BGE/Qwen cross-encoder reranker |
| | `transformers` | >=4.40.0, <5.0.0 | Transformer model backend |
| **Knowledge Graph** | `networkx` | >=3.2 | Graph construction and querying |
| **Web UI** | `gradio` | >=5.0.0 | Chat interface |
| **Utilities** | `rich` | >=13.0.0 | Terminal formatting |

### Models

| Role | Model | Parameters | Provider | Endpoint |
|------|-------|-----------|----------|----------|
| Embeddings | `mxbai-embed-large` | — | Ollama | `localhost:11434` |
| Chat LLM | `qwen3-235b` | 235B (MoE) | llama-server (llama.cpp) | `localhost:8080/v1` |
| Reranker | `Qwen/Qwen3-Reranker-4B` | 4B | Hugging Face / local | In-process (PyTorch) |

### Persistent Storage

| Artifact | Technology | Location | Size |
|----------|-----------|----------|------|
| Vector embeddings | ChromaDB (SQLite + HNSW) | `chroma_db/` | ~75 MB |
| Keyword index | Pickled BM25Okapi | `bm25_index.pkl` | ~8 MB |
| Knowledge graph | NetworkX GraphML | `knowledge_graph/` | Variable |

---

## Project Structure

```
my-first-rag/
├── config/
│   ├── __init__.py
│   └── settings.py            # Central configuration (all tunables)
├── core/
│   ├── embeddings.py           # OllamaEmbeddings wrapper
│   ├── llm.py                  # ChatOpenAI wrapper (for llama-server)
│   ├── store.py                # VectorStoreManager (ChromaDB)
│   └── query_service.py        # QueryService (routes all query modes)
├── modules/
│   ├── chunking.py             # PDF loading + text splitting strategies
│   ├── hybrid_search.py        # BM25Index + RRF fusion + HybridRetriever
│   ├── reranking.py            # Reranker + RerankingRetriever + MMRRerankingRetriever
│   ├── agentic.py              # Tool-calling agent with semantic + keyword tools
│   ├── knowledge_graph.py      # LLM-based triple extraction into NetworkX
│   └── multilingual.py         # Prompt templates
├── scripts/
│   ├── ingest.py               # CLI: document ingestion pipeline
│   ├── query.py                # CLI: interactive query loop
│   └── reset_store.py          # CLI: reset vector store / indices
├── data/                        # Input PDF documents
├── chroma_db/                   # Generated: ChromaDB persistent storage
├── bm25_index.pkl               # Generated: serialized BM25 index
├── app.py                       # Gradio web interface
├── requirements.txt             # Python dependencies
└── archive/
    └── prototype/rag.py         # Original single-file prototype (archived)
```

The codebase follows a three-layer architecture:

- **`config/`** — Centralised settings. Every tunable parameter lives in `settings.py`.
- **`core/`** — Infrastructure wrappers (embeddings, LLM, vector store) and the service layer (`QueryService`) that the CLI and web UI both depend on.
- **`modules/`** — Feature implementations (chunking strategies, retrieval algorithms, reranking, agentic reasoning). Each module is independent and can be used or bypassed through configuration.

---

## Core Components

### Configuration Layer

**File:** `config/settings.py`

All parameters are centralised in a single Python module. No module reads its own configuration — everything is imported from `settings.py`.

Key parameter groups:

```
Paths           PROJECT_ROOT, DATA_DIR, CHROMA_DIR, BM25_INDEX_PATH
Models          CHAT_MODEL, LLAMA_SERVER_BASE_URL, EMBED_MODEL, OLLAMA_BASE_URL
Chunking        CHUNK_SIZE (1000), CHUNK_OVERLAP (200)
                SECTION_CHUNK_SIZE (500), SECTION_CHUNK_OVERLAP (100), SECTION_SEPARATORS
Retrieval       RETRIEVER_K (4), HYBRID_VECTOR_WEIGHT (0.6), HYBRID_BM25_WEIGHT (0.4)
Reranking       RERANKER_ENABLED, RERANKER_MODEL, RERANKER_USE_FP16, RERANKER_BATCH_SIZE
MMR             MMR_LAMBDA_MULT (0.7), MMR_FETCH_K (50)
Agentic         MAX_AGENT_STEPS (5), AGENT_TEMPERATURE (0.1)
Web UI          WEB_HOST (0.0.0.0), WEB_PORT (7860), WEB_SHARE (False)
```

### Embedding Layer

**File:** `core/embeddings.py`

A thin factory function that returns a `langchain_ollama.OllamaEmbeddings` instance configured from settings. The embedding model (`mxbai-embed-large`, 1024 dimensions) runs on the Ollama server at port 11434.

```python
def get_embeddings(model=EMBED_MODEL) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
```

### LLM Layer

**File:** `core/llm.py`

Returns a `langchain_openai.ChatOpenAI` instance pointed at the llama-server's OpenAI-compatible endpoint. The project uses `ChatOpenAI` rather than `ChatOllama` to work around a nil-pointer bug in the Ollama Go server when running Qwen3 Mixture-of-Experts models.

Key parameters:
- `temperature`: 0.1 (low for factual regulatory answers)
- `request_timeout`: 120 seconds (needed for the large 235B model)

### Vector Store

**File:** `core/store.py`

`VectorStoreManager` wraps ChromaDB and provides:

- **Lazy loading** — Detects whether `chroma.sqlite3` exists on disk. If it does, the existing store is loaded without re-embedding.
- **Batch ingestion** — Documents are added in configurable batches (default 100) to avoid overwhelming the embedding server.
- **Retriever factory** — Returns LangChain-compatible retrievers with support for both standard similarity search and MMR (Maximal Marginal Relevance) diversity search.
- **Source tracking** — Stores `source_file`, `page`, and `chunk_strategy` in document metadata for provenance.

### Query Service

**File:** `core/query_service.py`

`QueryService` is the central service layer used by both the CLI (`scripts/query.py`) and the web UI (`app.py`). It provides three public methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `query_basic(question)` | `{answer, sources, mode}` | Vector search with optional reranking |
| `query_hybrid(question)` | `{answer, sources, mode}` | Vector + BM25 with RRF fusion |
| `query_agentic(question)` | `{answer, sources, reasoning_steps, mode}` | Multi-step agent reasoning |

Each method constructs a LangChain chain (or agent) on first call and caches it for subsequent queries. The chain pattern is:

```
{context: retriever | format_docs, question: RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
```

---

## Feature Modules

### Document Chunking

**File:** `modules/chunking.py`

Two chunking strategies, both using LangChain's `RecursiveCharacterTextSplitter`:

**Recursive (default)**
- Chunk size: 1000 characters, overlap: 200 characters
- Uses default separators (`\n\n`, `\n`, `. `, ` `)
- Suitable for general documents

**Section-aware**
- Chunk size: 500 characters, overlap: 100 characters
- Custom separator hierarchy designed for regulatory documents:
  ```
  \n## , \n### , \nSection , \nArticle , \nClause , \nAnnex , \n\n, \n, .
  ```
- Respects document structure by splitting at section boundaries first
- Metadata enriched with `source_file`, `page`, and `chunk_strategy`

The module also provides `load_directory()` which processes all PDFs in a folder, with optional skip-existing logic to avoid re-ingesting documents already in the store.

### Hybrid Search

**File:** `modules/hybrid_search.py`

Combines vector (semantic) and BM25 (keyword) retrieval using Reciprocal Rank Fusion (RRF).

**BM25Index class:**
- Builds an in-memory `BM25Okapi` index from document chunk text
- Tokenisation: simple lowercase whitespace splitting
- Serialised to disk via pickle for persistence across sessions
- Search returns top-k documents with non-zero BM25 scores

**Reciprocal Rank Fusion:**
```
RRF(d) = sum( weight_i / (k_constant + rank_i(d) + 1) )
```
- Default k_constant: 60
- Default weights: 0.6 vector, 0.4 BM25
- Deduplication by content hash

**HybridRetriever class:**
- Extends LangChain's `BaseRetriever` for seamless chain integration
- Fetches `k * 2` candidates from each source
- Fuses via RRF
- Optionally applies cross-encoder reranking before returning top-k

### Cross-Encoder Reranking

**File:** `modules/reranking.py`

Uses a cross-encoder model (`Qwen/Qwen3-Reranker-4B`) that jointly scores query-document pairs. Unlike bi-encoders (embedding models), cross-encoders consider the full interaction between query and document text for higher precision.

**Reranker class:**
- Singleton pattern to load the model once and reuse across queries
- Lazy loading: model is loaded on first use, not at import time
- Automatic device fallback: CUDA if available, otherwise CPU
- FP16 inference enabled on GPU for speed
- Batch processing (default 32 pairs per batch)
- Score normalisation via sigmoid to [0, 1] range
- Graceful degradation: returns original documents if reranking fails

**RerankingRetriever class:**
- LangChain `BaseRetriever` wrapper that fetches from a base retriever and reranks the results

**MMRRerankingRetriever class:**
- Three-stage pipeline: fetch candidates -> rerank with cross-encoder -> select diverse results with MMR
- Uses ChromaDB's native `max_marginal_relevance_search` on the reranked candidate set
- Preserves reranker scores in document metadata

### MMR Diversity Selection

MMR (Maximal Marginal Relevance) is integrated into the reranking pipeline and the vector store retriever. It selects documents that are both relevant to the query and diverse from each other.

```
MMR(d) = lambda * relevance(d) + (1 - lambda) * diversity(d)
```

- `lambda_mult = 0.7` (default): biased toward relevance with moderate diversity
- `fetch_k = 50`: number of initial candidates before MMR selection
- Final selection: top 4 diverse documents

### Agentic Reasoning

**File:** `modules/agentic.py`

Implements a tool-calling agent using LangChain's `create_tool_calling_agent` and `AgentExecutor`. The LLM decides which retrieval tools to call and when it has enough information to answer.

**Available tools:**

| Tool | Description |
|------|-------------|
| `semantic_search(query)` | Vector similarity search with optional reranking |
| `keyword_search(query)` | BM25 keyword search with optional reranking |

**Agent behaviour:**
- System prompt instructs the LLM to use semantic search for conceptual questions and keyword search for specific section/clause references
- The agent can call tools multiple times across up to `MAX_AGENT_STEPS` (5) iterations
- Each tool invocation retrieves and optionally reranks documents independently
- The agent synthesises information from all tool calls into a final answer
- Returns structured reasoning steps showing which tools were called and with what inputs

### Knowledge Graph

**File:** `modules/knowledge_graph.py`

An experimental module that extracts entity-relationship triples from document chunks using LLM calls.

- Extraction: the chat LLM is prompted to extract `(subject, predicate, object)` triples from each chunk
- Storage: triples are added to a NetworkX directed graph
- Persistence: saved as GraphML format
- Querying: finds incoming and outgoing relationships for a named entity

This module is slow in practice because it requires one LLM call per chunk.

---

## Retrieval Pipelines

### Basic Vector Search

```
Query -> Embed (mxbai-embed-large) -> ChromaDB similarity search (k=4)
      -> [Optional: Rerank with cross-encoder -> MMR diversity selection]
      -> Top 4 chunks -> Prompt template -> LLM -> Answer
```

### Hybrid Vector + BM25

```
Query ---+--- Vector similarity search (k=8) ----+
         |                                        |
         +--- BM25 keyword search (k=8) ---------+
                                                  |
                               Reciprocal Rank Fusion (0.6 / 0.4 weights)
                                                  |
                               [Optional: Cross-encoder reranking]
                                                  |
                               Top 4 chunks -> Prompt -> LLM -> Answer
```

### Reranking + MMR Pipeline

```
Query -> ChromaDB similarity search (fetch_k=50)
      -> Cross-encoder scores all 50 query-document pairs
      -> MMR diversity selection (lambda=0.7, select top 4)
      -> Prompt -> LLM -> Answer
```

### Agentic Multi-Step Pipeline

```
Query -> LLM Agent (with tool-calling capability)
      -> Agent decides: call semantic_search, keyword_search, or both
      -> Retrieves and reranks results per tool call
      -> Agent may iterate (up to 5 steps)
      -> Agent synthesises final answer from accumulated context
```

---

## Web Interface

**File:** `app.py`

Built with Gradio's `Blocks` API. The interface provides:

- **System status panel** (collapsible) — Shows vector store chunk count, BM25 index availability, reranker status, and model versions.
- **Query mode selector** — Radio buttons to switch between Basic, Hybrid, and Agentic modes.
- **Chat interface** — Message input with submit button, chat history display, and clear button.
- **Response formatting** — Answers include source citations (document name + page number) and collapsible reasoning steps in agentic mode.

`QueryService` is instantiated once at module level. The Gradio `respond` function delegates to the appropriate `query_*` method based on the selected mode and formats the structured response dictionary into markdown for display.

---

## External Services

The system requires two inference servers running at query time:

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| **Ollama** | 11434 | HTTP (Ollama API) | Serves the `mxbai-embed-large` embedding model |
| **llama-server** | 8080 | HTTP (OpenAI-compatible `/v1`) | Serves the `qwen3-235b` chat model via llama.cpp |

The reranker model (`Qwen3-Reranker-4B`) runs in-process using PyTorch and does not require a separate server.

---

## Data Flow

### Ingestion Pipeline

```
1. PDF files in data/ directory
2. PyPDFLoader extracts pages with text and page metadata
3. RecursiveCharacterTextSplitter chunks pages (recursive or section-aware)
4. Metadata enrichment: source_file, page, chunk_strategy
5. VectorStoreManager.add_documents() embeds chunks via Ollama and stores in ChromaDB
6. (Optional) BM25Index built from all chunk texts and pickled to disk
7. (Optional) Knowledge graph triples extracted via LLM and stored in NetworkX
```

### Query Pipeline

```
1. User submits question (CLI or web UI)
2. QueryService routes to the selected mode
3. Retriever fetches candidate documents (vector, BM25, or both)
4. (Optional) RRF fusion merges ranked lists
5. (Optional) Cross-encoder reranks candidates
6. (Optional) MMR selects diverse top-k
7. Retrieved chunks formatted into context string
8. Context + question inserted into prompt template
9. LLM generates answer
10. Response returned with source citations
```

---

## Key Design Decisions

**ChatOpenAI instead of ChatOllama** — The Qwen3-235B model is a Mixture-of-Experts architecture. The Ollama Go server has a known nil-pointer bug when handling MoE models. Running the model through llama-server and connecting via the OpenAI-compatible API (`ChatOpenAI`) avoids this issue.

**Singleton reranker** — The 4B-parameter cross-encoder model takes several seconds to load. The `Reranker` class uses a singleton pattern so the model is loaded once on first use and reused for all subsequent queries.

**BM25 index persistence** — The BM25 index is pickled to disk so it does not need to be rebuilt on every startup. This is important because building the index requires reading all document chunks from the vector store.

**RRF over score normalisation** — Reciprocal Rank Fusion merges ranked lists using rank positions rather than raw scores. This avoids the problem of normalising scores across methods that use fundamentally different scoring scales (cosine similarity vs BM25).

**Section-aware chunking** — Nuclear regulatory documents have strict hierarchical structure (Section, Article, Clause, Annex). The section-aware splitter uses custom separators that respect these boundaries, producing more coherent chunks than generic character splitting.

**Centralised configuration** — All parameters live in `config/settings.py`. This ensures that changing a model name, chunk size, or retrieval parameter only requires editing one file, regardless of how many modules consume that setting.

**Service layer pattern** — `QueryService` abstracts the chain/agent construction from the presentation layer. Both the CLI and Gradio UI import the same service class, ensuring consistent behaviour across interfaces.
