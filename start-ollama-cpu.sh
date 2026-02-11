#!/bin/bash
# Project-specific Ollama server running on CPU-only mode
# This won't affect the global Ollama service

export OLLAMA_NUM_GPU=0
export OLLAMA_HOST=127.0.0.1:11435  # Different port from default 11434

echo "Starting Ollama CPU-only server on port 11435..."
echo "This is isolated to this project and won't affect global Ollama."
ollama serve
