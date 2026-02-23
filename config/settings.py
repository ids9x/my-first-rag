"""
Central configuration for the RAG pipeline.
Change models, paths, and parameters here — not in individual modules.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
BM25_INDEX_PATH = PROJECT_ROOT / "bm25_index.pkl"
KNOWLEDGE_GRAPH_DIR = PROJECT_ROOT / "knowledge_graph"

# ── Models (swap these when you're ready to scale) ─────────────
# Chat LLM: llama.cpp server with Qwen3-235B (OpenAI-compatible API)
CHAT_MODEL = "qwen3-235b"                      # Model name for API calls
LLAMA_SERVER_BASE_URL = "http://localhost:8080/v1"  # llama-server endpoint
LLAMA_SERVER_API_KEY = "not-used"              # Required by OpenAI client but not validated

# Embeddings: Ollama instance for embedding model
EMBED_MODEL = "mxbai-embed-large"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Chunking ───────────────────────────────────────────────────
# Default recursive chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Section-aware chunking for structured standards (NQA-1, ASME, etc.)
SECTION_CHUNK_SIZE = 500
SECTION_CHUNK_OVERLAP = 100
SECTION_SEPARATORS = [
    "\n## ",          # Markdown H2
    "\n### ",         # Markdown H3
    "\nSection ",     # "Section 2.1 ..."
    "\nArticle ",     # "Article 5 ..."
    "\nClause ",      # "Clause 4.1 ..."
    "\nAnnex ",       # "Annex A ..."
    "\n\n",           # Double newline (paragraph break)
    "\n",             # Single newline
    ". ",             # Sentence boundary
    " ",              # Word boundary (last resort)
]

# ── Retrieval ──────────────────────────────────────────────────
RETRIEVER_K = 4                    # Number of chunks to retrieve
HYBRID_VECTOR_WEIGHT = 0.6        # 0.0 = pure BM25, 1.0 = pure vector
HYBRID_BM25_WEIGHT = 0.4

# ── Reranking ──────────────────────────────────────────────────
RERANKER_ENABLED = True                          # Disabled for CPU-only operation
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"        # Multilingual, ~568MB (CPU-friendly)
RERANKER_USE_FP16 = True                          # Faster inference on GPU
RERANKER_BATCH_SIZE = 32                          # Batch size for scoring
RERANKER_TOP_K = RETRIEVER_K                      # Final docs after reranking
RERANKER_FETCH_K = RETRIEVER_K * 2                # Retrieve more, rerank to top K
RERANKER_DEVICE = "cpu"                          # "cuda" or "cpu" (auto-fallback)
RERANKER_NORMALIZE_SCORES = True                  # Apply sigmoid to [0,1] range

# Fine-tuning support (future)
RERANKER_CUSTOM_MODEL_PATH = None                 # Path to fine-tuned model

# ── MMR Diversity Selection ────────────────────────────────────
# When reranking is enabled, MMR is automatically applied to select
# diverse documents from the reranked results
MMR_LAMBDA_MULT = 0.7                             # Relevance vs diversity tradeoff
                                                   # 1.0 = pure relevance (no diversity)
                                                   # 0.0 = pure diversity (less relevance)
                                                   # 0.5-0.7 recommended for cross-jurisdictional work
MMR_FETCH_K = 50                                  # Initial candidates before reranking + MMR

# ── Agentic RAG ───────────────────────────────────────────────
MAX_AGENT_STEPS = 5               # Max reasoning steps before forcing answer
AGENT_TEMPERATURE = 0.1           # Low temp for factual answers

# ── Knowledge Graph ────────────────────────────────────────────
KG_EXTRACTION_MODEL = CHAT_MODEL  # Can use a different model for extraction
KG_MAX_TRIPLES_PER_CHUNK = 10

# ── Collection naming ──────────────────────────────────────────
DEFAULT_COLLECTION = "documents"

# ── Web Interface ──────────────────────────────────────────────
WEB_HOST = "0.0.0.0"         # Listen on all network interfaces (accessible from network)
WEB_PORT = 7860              # Gradio default
WEB_SHARE = False            # Set True to create public Gradio link
