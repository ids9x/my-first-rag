# BUILD_PLAN.md — Map-Reduce & Multi-Strategy Patterns

## Goal

Add three new query strategies to the existing RAG pipeline, selectable via Gradio radio buttons alongside the current Basic / Hybrid / Agentic modes. The new modes are:

| # | Mode | One-liner |
|---|------|-----------|
| 1 | **Router** | LLM classifies the query and auto-dispatches to the best existing pipeline |
| 2 | **RAG → Map-Reduce** | Retrieve chunks, LLM-summarise each independently (map), then synthesise all summaries into a final answer (reduce) |
| 3 | **Parallel + Merge** | Run multiple retrieval strategies concurrently, deduplicate chunks, then produce a single merged answer |

Nothing in the existing codebase is modified destructively — all three modes are additive.

---

## Prerequisites

No new Python packages are required. Everything is built on top of LangChain primitives and the standard library already in `requirements.txt`. The `concurrent.futures` module (stdlib) handles parallelism.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                       Gradio UI (app.py)                 │
│  Radio: Basic │ Hybrid │ Agentic │ Router │ MapRed │ Par │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    QueryService     │
              │  core/query_service │
              ├─────────────────────┤
              │ query_basic()       │  ← existing
              │ query_hybrid()      │  ← existing
              │ query_agentic()     │  ← existing
              │ query_router()      │  ← NEW
              │ query_map_reduce()  │  ← NEW
              │ query_parallel()    │  ← NEW
              └────────┬────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   modules/        modules/     modules/
   router.py     map_reduce.py  parallel.py
```

Each new mode lives in its own module under `modules/` and exposes a clean function that `QueryService` calls. This matches the existing pattern where `modules/agentic.py`, `modules/hybrid_search.py`, etc. are consumed by `QueryService`.

---

## Phase 1 — Router Pattern

### What it does

The router uses a lightweight LLM call to classify the user's question into one of the existing retrieval strategies, then dispatches to that strategy automatically. The user doesn't need to know which pipeline is best — the LLM decides.

### Classification taxonomy

| Category | Routes to | Trigger signals |
|----------|-----------|-----------------|
| `factual_lookup` | Basic vector search | Short factual questions, definitions, specific clause lookups |
| `comparative` | Hybrid (vector + BM25) | Questions comparing sections, standards, requirements across documents |
| `exploratory` | Agentic multi-step | Open-ended analysis, multi-hop reasoning, "explain the implications of…" |

### New file: `modules/router.py`

```
modules/router.py
├── ROUTE_CLASSIFICATION_PROMPT   (str template)
├── classify_query()              → str ("factual_lookup" | "comparative" | "exploratory")
└── route_and_execute()           → dict {answer, sources, mode, routed_to, classification_reasoning}
```

**Implementation detail:**

1. `classify_query(question, llm)` sends the question to the LLM with a structured prompt asking it to output JSON: `{"category": "...", "reasoning": "..."}`.
2. The prompt includes few-shot examples for each category drawn from nuclear regulatory language.
3. `route_and_execute(question, query_service)` calls `classify_query`, then delegates to the matching `QueryService` method.
4. The response dict includes `routed_to` (which pipeline was chosen) and `classification_reasoning` (why), so the Gradio UI can show routing info in a collapsible details block.

**Prompt template sketch:**

```
You are a query classifier for a nuclear regulatory document retrieval system.
Classify the user question into exactly one category:

- factual_lookup: Direct factual questions, specific section/clause lookups, definitions.
- comparative: Comparing requirements across documents, standards, or sections.
- exploratory: Open-ended analysis, multi-step reasoning, implications.

Respond in JSON: {"category": "...", "reasoning": "one sentence explaining why"}

Question: {question}
```

### Config additions (`config/settings.py`)

```python
# Router
ROUTER_TEMPERATURE = 0.0          # deterministic classification
ROUTER_MAX_TOKENS = 150           # classification is short
```

### QueryService addition

```python
def query_router(self, question: str) -> dict:
    """Auto-route to best pipeline via LLM classification."""
    from modules.router import route_and_execute
    return route_and_execute(question, self)
```

---

## Phase 2 — RAG → Map-Reduce

### What it does

Instead of stuffing all retrieved chunks into a single prompt (which can lose detail with many chunks), map-reduce processes each chunk independently before synthesising:

```
Query
  │
  ▼
Retrieve N chunks (via hybrid retriever, reranked)
  │
  ├── Chunk 1 → LLM "map" call → Summary 1
  ├── Chunk 2 → LLM "map" call → Summary 2
  ├── Chunk 3 → LLM "map" call → Summary 3
  ...
  ├── Chunk N → LLM "map" call → Summary N
  │
  ▼
Collect all summaries
  │
  ▼
LLM "reduce" call → Final synthesised answer
```

### Why this matters for nuclear regulatory documents

Nuclear documents are long, dense, and contain precise language. Stuffing 4-8 chunks into one prompt often causes the LLM to fixate on the first or last chunk. Map-reduce forces the LLM to engage with every chunk individually, then combine them — producing more thorough answers that don't skip requirements buried in the middle.

### New file: `modules/map_reduce.py`

```
modules/map_reduce.py
├── MAP_PROMPT          (str template: "Given this document excerpt, extract information relevant to: {question}")
├── REDUCE_PROMPT       (str template: "Synthesise these summaries into a final answer for: {question}")
├── map_single_chunk()  → str (one chunk's summary)
├── map_all_chunks()    → list[str] (parallel map over all chunks)
└── map_reduce_query()  → dict {answer, sources, mode, map_summaries, chunk_count}
```

**Implementation detail:**

1. Retrieval uses the existing `HybridRetriever` (or `MMRRerankingRetriever` if reranker is enabled) to fetch `MAP_REDUCE_FETCH_K` chunks (default 8, more than the normal 4, because map-reduce can handle more context).
2. **Map phase**: Each chunk is sent to the LLM with the map prompt. Uses `concurrent.futures.ThreadPoolExecutor` for parallelism — the LLM server handles concurrent requests. Each map call extracts only the information relevant to the question from that chunk.
3. **Reduce phase**: All map summaries are concatenated (with source labels) and sent to the LLM with the reduce prompt to synthesise a final answer.
4. Source tracking is preserved: each map summary carries its chunk's `source_file` and `page` metadata.

**Map prompt sketch:**

```
You are analysing a nuclear regulatory document excerpt.

Question: {question}

Document excerpt (from {source_file}, page {page}):
{chunk_text}

Extract ONLY the information from this excerpt that is relevant to the question.
If this excerpt contains no relevant information, respond with "No relevant information."
Be precise and preserve technical terminology.
```

**Reduce prompt sketch:**

```
You are synthesising information from multiple nuclear regulatory document excerpts.

Question: {question}

Extracted information from {n} document excerpts:

{numbered_summaries}

Synthesise these into a comprehensive answer. Cite sources by their excerpt number.
If excerpts contain contradictory information, note the contradiction.
If no excerpts contained relevant information, say so clearly.
```

### Config additions (`config/settings.py`)

```python
# Map-Reduce
MAP_REDUCE_FETCH_K = 8            # chunks to retrieve (more than standard 4)
MAP_REDUCE_MAX_WORKERS = 4        # parallel map threads
MAP_REDUCE_MAP_MAX_TOKENS = 500   # map output per chunk
MAP_REDUCE_TEMPERATURE = 0.1      # factual extraction
```

### QueryService addition

```python
def query_map_reduce(self, question: str) -> dict:
    """Map-reduce: LLM processes each chunk independently, then synthesises."""
    from modules.map_reduce import map_reduce_query
    return map_reduce_query(question, self._get_retriever(), self._get_llm())
```

**Note:** `_get_retriever()` and `_get_llm()` are small internal helper methods to avoid duplicating the retriever/LLM setup logic. If these don't exist yet in `QueryService`, they should be extracted as part of this phase.

---

## Phase 3 — Parallel + Merge

### What it does

Runs multiple retrieval strategies at the same time, pools their results, deduplicates, and sends the merged chunk set to the LLM. This gives the broadest possible coverage — semantic similarity catches conceptual matches, BM25 catches exact keyword matches, and reranking catches high-precision matches.

```
Query
  │
  ├──── Thread 1: Basic vector search (k=6)     ──→ Chunks A
  ├──── Thread 2: Hybrid vector+BM25 (k=6)      ──→ Chunks B
  └──── Thread 3: MMR reranked search (k=6)      ──→ Chunks C
  │
  ▼
Merge A ∪ B ∪ C → Deduplicate by content hash → Rank by frequency + score
  │
  ▼
Top K merged chunks → Prompt → LLM → Final answer
```

### New file: `modules/parallel.py`

```
modules/parallel.py
├── run_strategy()         → list[Document] (runs one retrieval strategy)
├── merge_and_deduplicate()→ list[Document] (union + dedup + rank)
└── parallel_merge_query() → dict {answer, sources, mode, strategy_counts, total_unique_chunks}
```

**Implementation detail:**

1. Three retrieval strategies run concurrently via `ThreadPoolExecutor`:
   - Basic vector similarity (from `VectorStoreManager`)
   - Hybrid vector + BM25 (from `HybridRetriever`)
   - MMR reranking (from `MMRRerankingRetriever`, if reranker enabled; otherwise basic MMR from ChromaDB)
2. Each strategy fetches `PARALLEL_PER_STRATEGY_K` (default 6) chunks.
3. **Merge logic**:
   - Documents are deduplicated by content hash (same logic as `hybrid_search.py`).
   - Documents appearing in multiple strategy results get a frequency boost.
   - Final ranking: `combined_score = normalized_original_score + (0.1 × frequency_bonus)`.
   - Top `PARALLEL_FINAL_K` (default 6) documents are kept.
4. The merged set is formatted and sent to the LLM with the standard QA prompt.
5. Metadata tracks which strategies contributed each chunk (useful for debugging).

### Config additions (`config/settings.py`)

```python
# Parallel + Merge
PARALLEL_PER_STRATEGY_K = 6      # chunks per strategy
PARALLEL_FINAL_K = 6             # chunks after merge (can be more than standard 4)
PARALLEL_MAX_WORKERS = 3         # one thread per strategy
```

### QueryService addition

```python
def query_parallel(self, question: str) -> dict:
    """Parallel retrieval + merge: broadest coverage from multiple strategies."""
    from modules.parallel import parallel_merge_query
    return parallel_merge_query(question, self)
```

---

## Phase 4 — Gradio UI Updates

### File: `app.py`

**Changes:**

1. **Update radio buttons** — add the three new modes:

```python
mode = gr.Radio(
    choices=[
        "Basic", "Hybrid", "Agentic",
        "Router", "Map-Reduce", "Parallel + Merge"
    ],
    value="Basic",
    label="Query Mode"
)
```

2. **Update the `respond()` dispatch function:**

```python
def respond(message, history, mode):
    if mode == "Basic":
        result = query_service.query_basic(message)
    elif mode == "Hybrid":
        result = query_service.query_hybrid(message)
    elif mode == "Agentic":
        result = query_service.query_agentic(message)
    elif mode == "Router":
        result = query_service.query_router(message)
    elif mode == "Map-Reduce":
        result = query_service.query_map_reduce(message)
    elif mode == "Parallel + Merge":
        result = query_service.query_parallel(message)
    return format_response(result)
```

3. **Update `format_response()`** to handle new metadata fields:

- **Router mode**: Show a collapsible "Routing decision" block with the classification and which pipeline was selected.
- **Map-Reduce mode**: Show a collapsible "Map summaries" block listing each chunk's individual summary before the final synthesised answer.
- **Parallel + Merge mode**: Show a collapsible "Strategy contributions" block showing how many unique chunks each strategy contributed.

4. **Update system status panel** to show the three new modes as available.

---

## Phase 5 — Config & Settings Update

All new settings go into `config/settings.py`, grouped under clear comment headers:

```python
# =============================================================================
# Router Pattern
# =============================================================================
ROUTER_TEMPERATURE = 0.0
ROUTER_MAX_TOKENS = 150

# =============================================================================
# Map-Reduce
# =============================================================================
MAP_REDUCE_FETCH_K = 8
MAP_REDUCE_MAX_WORKERS = 4
MAP_REDUCE_MAP_MAX_TOKENS = 500
MAP_REDUCE_TEMPERATURE = 0.1

# =============================================================================
# Parallel + Merge
# =============================================================================
PARALLEL_PER_STRATEGY_K = 6
PARALLEL_FINAL_K = 6
PARALLEL_MAX_WORKERS = 3
```

---

## Implementation Order

Build and test each phase independently. The order below is intentional — each phase builds on confidence from the previous one.

### Step 1: Router (simplest — no new retrieval logic)

1. Create `modules/router.py` with `classify_query()` and `route_and_execute()`.
2. Add `query_router()` to `QueryService`.
3. Add Router to Gradio radio and dispatch.
4. **Test**: Ask questions of varying types and verify the router picks the right pipeline. Check the classification reasoning makes sense.

### Step 2: Map-Reduce (new retrieval pattern, moderate complexity)

1. Create `modules/map_reduce.py` with map/reduce prompts and functions.
2. Extract `_get_retriever()` / `_get_llm()` helpers in `QueryService` if needed.
3. Add `query_map_reduce()` to `QueryService`.
4. Add Map-Reduce to Gradio radio and dispatch.
5. **Test**: Ask a complex question that spans multiple documents (e.g., "What are all the requirements for quality assurance in nuclear facility design?"). Compare the map-reduce answer with the basic mode answer — map-reduce should be more thorough.

### Step 3: Parallel + Merge (uses all existing retrievers concurrently)

1. Create `modules/parallel.py` with parallel execution and merge logic.
2. Add `query_parallel()` to `QueryService`.
3. Add Parallel + Merge to Gradio radio and dispatch.
4. **Test**: Ask a question that has both conceptual and keyword-specific aspects. Check that the strategy contributions show chunks from multiple strategies. Verify deduplication works (no repeated chunks in the final answer).

### Step 4: UI polish

1. Update `format_response()` for all three new modes.
2. Update the system status panel.
3. Test all six modes end-to-end through Gradio.

---

## File Change Summary

| File | Action | What changes |
|------|--------|-------------|
| `modules/router.py` | **CREATE** | Router classification + dispatch |
| `modules/map_reduce.py` | **CREATE** | Map and reduce prompts + orchestration |
| `modules/parallel.py` | **CREATE** | Parallel retrieval + merge + dedup |
| `core/query_service.py` | **EDIT** | Add `query_router()`, `query_map_reduce()`, `query_parallel()`, helper methods |
| `config/settings.py` | **EDIT** | Add Router / Map-Reduce / Parallel config blocks |
| `app.py` | **EDIT** | Update radio choices, dispatch, response formatting |

**No existing files are deleted or fundamentally restructured.**

---

## Updated Project Structure (after all phases)

```
my-first-rag/
├── config/
│   ├── __init__.py
│   └── settings.py                # + Router, Map-Reduce, Parallel settings
├── core/
│   ├── embeddings.py
│   ├── llm.py
│   ├── store.py
│   └── query_service.py           # + query_router, query_map_reduce, query_parallel
├── modules/
│   ├── chunking.py
│   ├── hybrid_search.py
│   ├── reranking.py
│   ├── agentic.py
│   ├── knowledge_graph.py
│   ├── multilingual.py
│   ├── router.py                  # NEW
│   ├── map_reduce.py              # NEW
│   └── parallel.py                # NEW
├── scripts/
│   ├── ingest.py
│   ├── query.py
│   └── reset_store.py
├── data/
├── chroma_db/
├── bm25_index.pkl
├── app.py                         # Updated: 6 modes
├── requirements.txt
├── TECHNICAL.md
├── BUILD_PLAN.md                  # This file
├── CLAUDE.md                      # Claude Code instructions
└── archive/
    └── prototype/rag.py
```

---

## Error Handling & Edge Cases

Each new module must handle these gracefully:

- **Router**: If classification JSON parsing fails, fall back to Hybrid mode (best general-purpose) and log a warning.
- **Map-Reduce**: If a map call fails for one chunk, skip it and continue with the others. If all map calls fail, fall back to standard stuffed-context approach. If retriever returns 0 chunks, return a "no documents found" response.
- **Parallel**: If one strategy thread fails (e.g., BM25 index missing), continue with the others. If all strategies fail, return an error. Deduplication must handle missing metadata gracefully.
- **Timeouts**: Map-reduce map calls use the existing 120-second timeout per LLM call. Parallel strategies inherit existing retriever timeouts. Add a total timeout for the parallel phase (configurable, default 180 seconds).

---

## Performance Notes

- **Router** adds one short LLM call (~1-3 seconds with Qwen3-235B) before the actual retrieval. Acceptable overhead for automatic pipeline selection.
- **Map-Reduce** makes N+1 LLM calls (N maps + 1 reduce). With `MAP_REDUCE_FETCH_K=8` and 4 parallel workers, the map phase takes roughly 2× the time of a single LLM call (since 4 run concurrently). The reduce call adds another ~5-10 seconds. Total: ~15-25 seconds. This is the slowest mode by design.
- **Parallel + Merge** adds retrieval latency (3 strategies) but they run concurrently, so wall-clock time ≈ the slowest single strategy + merge overhead. Only one LLM generation call at the end. Total: comparable to Hybrid mode.
- Your Blackwell GPU with 128GB vRAM can handle concurrent LLM inference requests — llama-server supports request batching natively.
