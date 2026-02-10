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
CHAT_MODEL = "gemma3:4b"           # Small for testing; later: "nemotron", "qwen2.5:32b"
EMBED_MODEL = "nomic-embed-text"   # 768-dim, good multilingual support
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

# ── Agentic RAG ───────────────────────────────────────────────
MAX_AGENT_STEPS = 5               # Max reasoning steps before forcing answer
AGENT_TEMPERATURE = 0.1           # Low temp for factual answers

# ── Knowledge Graph ────────────────────────────────────────────
KG_EXTRACTION_MODEL = CHAT_MODEL  # Can use a different model for extraction
KG_MAX_TRIPLES_PER_CHUNK = 10

# ── Collection naming ──────────────────────────────────────────
DEFAULT_COLLECTION = "documents"
