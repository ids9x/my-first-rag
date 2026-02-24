# CLAUDE.md — Claude Code Project Instructions

## Project Identity

This is **my-first-rag**, a modular Retrieval-Augmented Generation pipeline for nuclear regulatory documents. It runs on an ASUS GX10 Ascent (single NVIDIA Blackwell GPU, 128GB vRAM) with Ubuntu. The developer connects via SSH from VS Code on Windows 11 and uses Claude Code exclusively for all development.

**The developer is a novice programmer.** Write clear, well-commented code. Prefer explicit over clever. When in doubt, add a comment explaining *why*, not just *what*.

---

## Critical Rules

1. **Never delete or overwrite existing working code.** All changes are additive. If refactoring a function, keep the original working path intact and add the new one alongside it.
2. **Never modify `config/settings.py` structure** — only append new settings at the bottom in clearly labelled sections.
3. **Always preserve the existing six-mode Gradio interface** (Basic, Hybrid, Agentic, Router, Map-Reduce, Parallel + Merge). Never remove a mode.
4. **Test one thing at a time.** After writing a module, test it in isolation before wiring it into `QueryService` or `app.py`.
5. **All imports from modules/ are lazy** (inside function bodies in `QueryService`), not at file top-level. This is the existing pattern — follow it.

---

## System Architecture

```
Windows 11 (VS Code + Claude Code)
    │ SSH
    ▼
ASUS GX10 (Ubuntu, Blackwell GPU, 128GB vRAM)
    ├── Ollama (port 11434)         → mxbai-embed-large embeddings
    ├── llama-server (port 8080)    → qwen3-235b chat model
    ├── ChromaDB (on-disk)          → vector store
    ├── BM25 index (pickle)         → keyword search
    └── Gradio (port 7860)          → web UI
```

---

## Project Layout

```
my-first-rag/
├── config/settings.py          # ALL tunables live here. Single source of truth.
├── core/
│   ├── embeddings.py           # OllamaEmbeddings factory
│   ├── llm.py                  # ChatOpenAI factory (→ llama-server)
│   ├── store.py                # VectorStoreManager (ChromaDB wrapper)
│   └── query_service.py        # Central service layer (6 query methods)
├── modules/
│   ├── chunking.py             # PDF loading + text splitting
│   ├── hybrid_search.py        # BM25 + RRF fusion + HybridRetriever
│   ├── reranking.py            # Cross-encoder reranker + MMR
│   ├── agentic.py              # Tool-calling agent
│   ├── knowledge_graph.py      # NetworkX triple extraction
│   ├── multilingual.py         # Prompt templates
│   ├── router.py               # LLM query classifier + dispatch
│   ├── map_reduce.py           # Map-reduce over retrieved chunks
│   └── parallel.py             # Parallel retrieval + merge
├── scripts/
│   ├── ingest.py               # CLI: document ingestion
│   ├── query.py                # CLI: interactive query loop
│   └── reset_store.py          # CLI: reset vector store
├── app.py                      # Gradio web UI (6 modes)
├── data/                       # Input PDFs
├── chroma_db/                  # ChromaDB storage (generated)
├── bm25_index.pkl              # BM25 index (generated)
├── requirements.txt
├── BUILD_PLAN.md               # Implementation plan
└── CLAUDE.md                   # This file
```

---

## Coding Conventions

### Style

- Python 3.12. Use type hints on all function signatures.
- Docstrings on every public function (Google style, keep short).
- Max line length: 100 characters (soft limit, don't break readability for it).
- Use `pathlib.Path` for file paths, not string concatenation.
- f-strings for formatting, not `.format()` or `%`.

### Imports

- Standard library first, then third-party, then project-internal. Blank line between each group.
- Inside `QueryService`, module imports are **lazy** (inside method bodies):
  ```python
  def query_router(self, question: str) -> dict:
      from modules.router import route_and_execute  # lazy import
      return route_and_execute(question, self)
  ```
- This avoids import-time crashes when optional dependencies (reranker model, BM25 index) aren't available.

### Configuration

- **Every** tunable parameter lives in `config/settings.py`.
- Modules import from settings: `from config.settings import MAP_REDUCE_FETCH_K`.
- Never hardcode URLs, ports, model names, temperatures, chunk sizes, or k-values in module files.

### Error Handling

- Wrap external calls (LLM, retriever, reranker) in try/except.
- On failure, log with `rich.console.Console().print(f"[red]Error: ...[/red]")`.
- Degrade gracefully: if one component fails, return partial results with a warning, don't crash.
- Return consistent response dicts from every `query_*` method.

### Return Value Contract

Every `query_*` method in `QueryService` returns a dict with at minimum:
```python
{
    "answer": str,       # The LLM's response
    "sources": list,     # List of {source_file, page} dicts
    "mode": str          # Human-readable mode name
}
```

Additional keys per mode:
- **Router**: `"routed_to"` (str), `"classification_reasoning"` (str)
- **Map-Reduce**: `"map_summaries"` (list[str]), `"chunk_count"` (int)
- **Parallel + Merge**: `"strategy_counts"` (dict), `"total_unique_chunks"` (int)
- **Agentic**: `"reasoning_steps"` (list)

---

## Key Implementation Patterns

### LLM Access

Always use the factory in `core/llm.py`:
```python
from core.llm import get_llm
llm = get_llm()  # Returns ChatOpenAI pointed at llama-server:8080
```

For custom temperature (e.g., router classification):
```python
from core.llm import get_llm
from config.settings import ROUTER_TEMPERATURE
llm = get_llm(temperature=ROUTER_TEMPERATURE)
```

If `get_llm()` doesn't accept kwargs yet, modify it to pass through `**kwargs` to `ChatOpenAI`.

### Retriever Access

The existing `QueryService` constructs retrievers internally. For new modules that need a retriever, either:
1. Accept the retriever as a parameter (preferred — keeps modules testable).
2. Or use `VectorStoreManager` and `HybridRetriever` directly.

Pattern:
```python
def map_reduce_query(question: str, retriever, llm) -> dict:
    """Module function — receives dependencies, doesn't construct them."""
    docs = retriever.invoke(question)
    # ...
```

### Parallel Execution

Use `concurrent.futures.ThreadPoolExecutor` for I/O-bound parallelism (LLM calls, retriever calls):
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(fn, arg): label for arg, label in tasks}
    results = {}
    for future in as_completed(futures):
        label = futures[future]
        try:
            results[label] = future.result(timeout=120)
        except Exception as e:
            console.print(f"[yellow]Warning: {label} failed: {e}[/yellow]")
```

### Document Deduplication

Reuse the content-hash approach from `modules/hybrid_search.py`:
```python
seen = set()
unique_docs = []
for doc in all_docs:
    content_hash = hash(doc.page_content)
    if content_hash not in seen:
        seen.add(content_hash)
        unique_docs.append(doc)
```

### Prompt Templates

Use LangChain's `ChatPromptTemplate`:
```python
from langchain_core.prompts import ChatPromptTemplate

MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are analysing a nuclear regulatory document excerpt."),
    ("human", "Question: {question}\n\nExcerpt from {source}:\n{chunk}\n\nExtract relevant information.")
])
```

---

## The Three New Modules — Quick Reference

### modules/router.py

**Purpose**: Classify query type → dispatch to best existing pipeline.

Key functions:
- `classify_query(question: str, llm) -> dict` — Returns `{"category": str, "reasoning": str}`.
- `route_and_execute(question: str, query_service) -> dict` — Classifies then calls the matched `query_*` method.

Classification categories: `factual_lookup` → Basic, `comparative` → Hybrid, `exploratory` → Agentic.

Fallback: If JSON parsing fails, default to Hybrid.

### modules/map_reduce.py

**Purpose**: Process each retrieved chunk independently, then synthesise.

Key functions:
- `map_single_chunk(question: str, doc, llm) -> str` — LLM extracts relevant info from one chunk.
- `map_all_chunks(question: str, docs: list, llm, max_workers: int) -> list[str]` — Parallel map.
- `map_reduce_query(question: str, retriever, llm) -> dict` — Full pipeline.

Uses `MAP_REDUCE_FETCH_K` (8) chunks, `MAP_REDUCE_MAX_WORKERS` (4) threads.

### modules/parallel.py

**Purpose**: Run multiple retrieval strategies concurrently, merge results.

Key functions:
- `run_strategy(name: str, retriever, question: str) -> list[Document]` — Runs one strategy.
- `merge_and_deduplicate(results: dict[str, list[Document]]) -> list[Document]` — Union + dedup + rank.
- `parallel_merge_query(question: str, query_service) -> dict` — Full pipeline.

Strategies: basic vector, hybrid, MMR reranked. Each fetches `PARALLEL_PER_STRATEGY_K` (6) chunks.

---

## Gradio UI Structure

The Gradio interface in `app.py` uses `gr.Blocks`. The current structure:
- System status panel (collapsible)
- Radio buttons for mode selection
- Chat interface with submit/clear

**When updating, preserve this structure.** Only modify:
1. The `choices` list in `gr.Radio` to add new modes.
2. The `respond()` function to dispatch to new modes.
3. The `format_response()` helper to render new metadata fields.

New mode metadata should display in collapsible `<details>` blocks in the markdown response, matching the existing pattern for agentic reasoning steps.

---

## External Services Check

Before running any query, verify services are up:
- **Ollama**: `curl http://localhost:11434/api/tags` should return model list.
- **llama-server**: `curl http://localhost:8080/v1/models` should return model info.
- **Gradio**: Starts on `0.0.0.0:7860` (accessible from Windows browser via SSH tunnel or direct IP).

---

## Testing Approach

The developer tests interactively via Gradio and CLI. There is no automated test suite (yet). When building new features:

1. **Module-level test**: After writing `modules/router.py`, test it in isolation:
   ```bash
   cd ~/my-first-rag
   python -c "
   from core.llm import get_llm
   from modules.router import classify_query
   result = classify_query('What is the definition of safety class?', get_llm())
   print(result)
   "
   ```
2. **Service-level test**: After wiring into `QueryService`:
   ```bash
   python -c "
   from core.query_service import QueryService
   qs = QueryService()
   result = qs.query_router('Compare NQA-1 and ISO 19443 requirements')
   print(result)
   "
   ```
3. **UI test**: Run `python app.py`, open browser, test each mode via Gradio.

---

## Common Pitfalls

- **Do not use `ChatOllama` for the chat model.** There is a known bug with MoE models in the Ollama Go server. Use `ChatOpenAI` pointed at llama-server. See `core/llm.py`.
- **The reranker may not be loaded.** Always check `RERANKER_ENABLED` in settings before using it. If disabled, skip reranking steps gracefully.
- **BM25 index may not exist.** If `bm25_index.pkl` is missing, hybrid search and any strategy that depends on it should degrade to vector-only.
- **LLM calls are slow.** Qwen3-235B on a single GPU takes 5-15 seconds per response. Design for this: show progress indicators in Gradio, use parallel execution where possible.
- **JSON parsing from LLM output.** The LLM may wrap JSON in markdown code fences. Always strip ` ```json ` and ` ``` ` before parsing. Use a helper:
  ```python
  import json
  import re

  def parse_llm_json(text: str) -> dict:
      """Parse JSON from LLM output, stripping markdown fences if present."""
      cleaned = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`')
      return json.loads(cleaned)
  ```

---

## Build Plan Reference

See `BUILD_PLAN.md` for the full phased implementation plan with architecture diagrams, prompt templates, config values, and the recommended build order:

1. Router (simplest — wraps existing pipelines)
2. Map-Reduce (new retrieval pattern)
3. Parallel + Merge (concurrent multi-strategy)
4. UI polish

Always implement and test one phase completely before starting the next.
