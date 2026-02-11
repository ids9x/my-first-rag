#!/bin/bash
# Start Ollama server on CPU-only mode on port 11435

echo "Starting Ollama server on port 11435 (CPU-only)..."
OLLAMA_HOST=0.0.0.0:11435 ollama serve
