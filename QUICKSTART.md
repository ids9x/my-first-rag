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

## 5. Start Gradio UI

```bash
python app.py
# Opens at http://127.0.0.1:7860
```

## 6. Shut Down

```bash
# Stop Gradio: Ctrl+C in its terminal

# Stop llama-server (frees VRAM)
pkill -f llama-server

# Stop Ollama
pkill -f "ollama serve"

# Verify GPU memory is freed
nvidia-smi
```
