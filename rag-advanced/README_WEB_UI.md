# 🌐 RAG Web Interface - User Guide

A simple web-based chat interface for your Nuclear Regulatory RAG system, powered by Gradio.

---

## 📋 Quick Start

### 1. Prerequisites

Before starting the web interface, ensure:

✅ **Ollama is running:**
```bash
ollama serve
```

✅ **Vector store exists:**
```bash
python -m scripts.ingest
```

✅ **(Optional) BM25 index built** (for Hybrid mode):
```bash
python -m scripts.ingest --build-bm25
```

---

### 2. Start the Web Interface

From the `rag-advanced` directory, run:

```bash
python app.py
```

You'll see output like:
```
============================================================
🚀 Starting Gradio web interface...
📍 URL: http://0.0.0.0:7860
============================================================
```

**The app is now running!**

---

### 3. Access the Interface

#### **From the same machine:**
Open your browser and go to:
```
http://localhost:7860
```

#### **From another computer on your network:**
Use the server's IP address:
```
http://192.168.1.45:7860
```
*(Replace `192.168.1.45` with your server's actual IP)*

---

## 💬 Using the Interface

### Main Components

1. **📊 System Status** (collapsible panel at top)
   - Shows vector store status and chunk count
   - Indicates if BM25 index is available
   - Shows reranker status (enabled/disabled)

2. **🔘 Query Mode Selector**
   Choose from three modes:
   - **Basic (Vector)**: Semantic search using embeddings - best for conceptual questions
   - **Hybrid (Vector+BM25)**: Combines semantic + keyword search - best for specific sections or terms
   - **Agentic (Multi-step)**: AI agent with reasoning tools - best for complex questions requiring multiple steps

3. **💬 Chat Interface**
   - Type your question in the text box
   - View conversation history above
   - Click example questions to try them out

### Example Queries

**Basic Mode:**
```
What are the QA requirements for nuclear facilities?
```

**Hybrid Mode:**
```
Explain NQA-1 Section 18
```

**Agentic Mode:**
```
What is the difference between QA Level 1 and Level 2?
```

---

## 📚 Understanding Responses

### Sources
Each answer includes sources at the bottom:
```
📚 Sources:
1. NQA-1.pdf, page 23
2. ASME-standard.pdf, page 45
```

### Reasoning Steps (Agentic Mode Only)
Click "🧠 Show Reasoning Steps" to see:
- Which tools the agent used
- What information it searched for
- How it arrived at the answer

---

## 🛑 Stopping the Web Interface

### Method 1: Terminal (if running in foreground)
Press `Ctrl+C` in the terminal where you started the app.

### Method 2: Kill the process
```bash
pkill -f "python app.py"
```

### Method 3: Find and kill by PID
```bash
# Find the process ID
ps aux | grep "python app.py" | grep -v grep

# Kill using the PID (second column)
kill <PID>
```

---

## ⚙️ Configuration

Edit `config/settings.py` to customize:

```python
# ── Web Interface ──────────────────────────────────────────────
WEB_HOST = "0.0.0.0"         # "127.0.0.1" for localhost only
WEB_PORT = 7860              # Change to use a different port
WEB_SHARE = False            # Set True for public Gradio link
```

**After changing settings, restart the app.**

---

## 🔧 Troubleshooting

### Problem: "Vector store not found"
**Solution:**
```bash
python -m scripts.ingest
```

### Problem: "BM25 index not found" (Hybrid mode)
**Solution:**
```bash
python -m scripts.ingest --build-bm25
```

### Problem: "Cannot connect to Ollama"
**Solution:**
Make sure Ollama is running:
```bash
ollama serve
```

### Problem: Can't access from another computer
**Solutions:**

1. **Check the config** - Ensure `WEB_HOST = "0.0.0.0"` in `config/settings.py`

2. **Check firewall:**
   ```bash
   sudo ufw allow 7860/tcp
   ```

3. **Verify IP address:**
   ```bash
   ip addr show | grep "inet "
   ```

4. **Test locally first:**
   ```bash
   curl http://localhost:7860
   ```

### Problem: Port 7860 already in use
**Solution:**
Either kill the existing process or change the port in `config/settings.py`:
```python
WEB_PORT = 7861  # Use a different port
```

---

## 🚀 Advanced Usage

### Run in Background
```bash
nohup python app.py > /dev/null 2>&1 &
```

### Check if Running
```bash
ps aux | grep "python app.py" | grep -v grep
```

### View Logs (if using nohup)
```bash
tail -f nohup.out
```

---

## 📊 Query Modes Comparison

| Mode | Best For | Requires |
|------|----------|----------|
| **Basic** | Conceptual questions, general topics | Vector store |
| **Hybrid** | Specific sections, exact terms, clause references | Vector store + BM25 |
| **Agentic** | Complex questions, comparisons, multi-step reasoning | Vector store (+ optional BM25) |

---

## 🆘 Getting Help

### Check System Status
1. Open the "📊 System Status" accordion in the web interface
2. Verify all components show ✅

### CLI Alternative
If the web interface isn't working, use the CLI:
```bash
python -m scripts.query              # Basic mode
python -m scripts.query --mode hybrid    # Hybrid mode
python -m scripts.query --mode agentic   # Agentic mode
```

---

## 📝 Notes

- **Chat history** persists during your session but is cleared when you close the browser
- **Multiple users** can access simultaneously (requests are queued)
- **Response time** depends on your hardware and the complexity of the query
- **Reranking** is currently disabled (CPU-only mode) - enable in `config/settings.py` if you have GPU support

---

## 🎯 Tips for Best Results

1. **Be specific**: "Explain NQA-1 Section 18" works better than "Tell me about quality"
2. **Use Hybrid mode** for technical terms or specific sections
3. **Use Agentic mode** for questions requiring reasoning or comparison
4. **Check sources** to verify the answer is based on your documents
5. **Try different phrasings** if you don't get the expected result

---

**Enjoy your RAG web interface! 🎉**

For issues or questions, refer to the main project README or check the troubleshooting section above.
