# Quick Start

## 1. Start Servers

```bash
# Terminal 1: Ollama (embeddings)
ollama serve &

# Terminal 2: llama-server (chat LLM) — from the project root
./start-llama-server.sh
# Wait for "listening on 0.0.0.0:8081"
```

Verify both are running:

```bash
curl http://localhost:11434/api/tags       # Ollama
curl http://localhost:8081/v1/models       # llama-server
```

## 2. Activate Environment

```bash
cd ~/my-first-rag
source .venv/bin/activate
```

## 3. Ingest Documents

Place files in `data/` (PDF, DOCX, XLSX, .eml, .msg, TXT), then:

```bash
python -m scripts.ingest --strategy section --build-bm25
```

## 4. Re-ingest After Adding New Documents

```bash
# Reset vector store + BM25, then re-ingest everything
python -m scripts.reset_store --all
python -m scripts.ingest --strategy section --build-bm25
```

Verify chunk count:

```bash
python -c "from core.store import VectorStoreManager; m = VectorStoreManager(); print(f'Chunks: {m.get_store()._collection.count()}')"
```

## 5. Batch Queries

For running many questions at once without the UI, use the batch workflow.

### a) Edit `questions.yaml`

```yaml
settings:
  preset: "Nuclear Technical"     # or "DRB Expert"
  output_dir: results/

questions:
  - What are the primary containment boundary specifications?
  - Compare NQA-1 and ISO 19443 quality requirements
  - What are the consequences of non-conformance in nuclear procurement?

  # You can also pin a specific mode for any question:
  # - question: Trace the full safety analysis methodology
  #   mode: agentic
```

### b) Two-step workflow (classify, review, then run)

```bash
# Step 1: LLM classifies each question → updates questions.yaml with per-question modes
python -m scripts.batch_query classify questions.yaml

# Review/edit the classified YAML if you want to override any modes

# Step 2: Execute all questions → writes JSON + CSV to results/
python -m scripts.batch_query run questions.yaml
```

### c) One-shot (classify + run combined)

```bash
python -m scripts.batch_query auto questions.yaml
```

### d) Override mode for all questions

```bash
python -m scripts.batch_query run questions.yaml --mode hybrid
```

Results are written to `results/batch_results_<timestamp>.json` and `.csv`.

Available modes: `basic`, `hybrid`, `agentic`, `router`, `map_reduce`, `parallel`.

## 6. Start Gradio UI

```bash
python app.py
# Opens at http://127.0.0.1:7860
```

## 7. Shut Down

```bash
# Stop Gradio: Ctrl+C in its terminal

# Stop llama-server (frees VRAM)
pkill -f llama-server

# Stop Ollama
pkill -f "ollama serve"

# Verify GPU memory is freed
nvidia-smi
```
