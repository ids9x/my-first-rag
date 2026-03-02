#!/bin/bash
# Start llama-server for the RAG pipeline (Qwen3-235B model)
# Uses port 8081 to avoid conflicts with other services on 8080

PORT=8081

# Kill any existing llama-server on this port
if fuser ${PORT}/tcp >/dev/null 2>&1; then
  echo "Port ${PORT} is in use — stopping existing process..."
  fuser -k ${PORT}/tcp 2>/dev/null
  sleep 1
fi

cd /home/ids9x/llama.cpp
./build/bin/llama-server \
  --model /home/ids9x/qwen3-235b/UD-Q3_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q3_K_XL-00001-of-00003.gguf \
  --host 0.0.0.0 \
  --port ${PORT} \
  --ctx-size 32768 \
  --parallel 1 \
  --n-gpu-layers -1 \
  --threads 8 \
  --batch-size 512 \
  2>&1 | tee llama-server.log
