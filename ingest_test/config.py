"""Central configuration for the Docling ingest pipeline."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent / ".chromadb"
COLLECTION_NAME = "nuclear_docs"

# Optional: local path to pre-downloaded Docling model artifacts
# Set this for offline / air-gapped operation on your DGX Spark
# Download first with: docling-tools models download --dir ./models
ARTIFACTS_PATH = None  # e.g. Path("/home/user/models/docling")

# ── Converter settings ─────────────────────────────────
DO_OCR = True                     # Enable for scanned IAEA docs
DO_TABLE_STRUCTURE = True         # Critical for compliance matrices
TABLE_CELL_MATCHING = True        # Map structure back to PDF cells
TABLE_MODE = "ACCURATE"           # "FAST" or "ACCURATE"
OCR_LANG = ["en", "de"]           # Your DE/EN document mix
NUM_THREADS = 4                   # CPU threads for accelerator

# ── Chunker settings ──────────────────────────────────
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"  # tokenizer alignment
CHUNK_MAX_TOKENS = 512            # Match your embedding model's limit
CHUNK_MERGE_PEERS = True          # Merge small adjacent chunks in same section

# ── Query settings ─────────────────────────────────────
TOP_K = 5