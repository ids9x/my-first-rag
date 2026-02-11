#!/bin/bash
# Stop Ollama server

echo "Stopping Ollama server..."
pkill -f "ollama serve" && echo "Ollama stopped successfully" || echo "No Ollama process found"