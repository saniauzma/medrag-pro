# config.py
# ---------
# Central settings for MedRAG Pro.
#
# How it works:
#   1. pydantic-settings reads your .env file automatically
#   2. Every setting has a default — .env overrides the default
#   3. Import `settings` anywhere: from config import settings
#   4. Access values like: settings.llm_model, settings.qdrant_port
#
# Why pydantic-settings?
#   - Type validation — if you put "abc" for a port number, it errors immediately
#   - Auto-casting — "true" in .env becomes True (bool) in Python
#   - One place to change config — no hunting through files

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── Project ──────────────────────────────────────────────
    # Basic metadata — not critical but good practice
    project_name: str = "MedRAG Pro"
    version: str = "0.1.0"

    # debug=True enables extra logging throughout the app
    debug: bool = False

    # ── Paths ─────────────────────────────────────────────────
    # Path() gives us OS-independent path handling
    # On Windows: Path("data/pdfs") works fine — no need for backslashes
    data_dir: Path = Path("data")
    pdf_dir: Path = Path("data/pdfs")
    index_dir: Path = Path("data/indexes")   # BM25 index saved here
    cache_dir: Path = Path("data/cache")     # extracted images cached here

    # ── Ollama ───────────────────────────────────────────────
    # Ollama exposes a REST API at this URL when running
    ollama_base_url: str = "http://localhost:11434"

    # Text LLM — handles question answering
    llm_model: str = "llama3.2:3b"

    # Vision LLM — handles image captioning from PDFs
    vlm_model: str = "llava-phi3"

    # temperature=0.0 means fully deterministic output
    # Critical for medical Q&A — we want consistent, not creative answers
    llm_temperature: float = 0.0

    llm_max_tokens: int = 1024

    # ── Embeddings ───────────────────────────────────────────
    # BAAI/bge-m3 is the best open-source embedding model as of 2024
    # 1024-dim output, 8192 token context window
    embedding_model: str = "BAAI/bge-m3"

    # "cuda" uses your NVIDIA GPU — much faster than CPU
    # Change to "cpu" if you don't have a GPU
    embedding_device: str = "auto"

    # How many chunks to embed in one GPU batch
    # Larger = faster but uses more VRAM. 32 is safe for 8GB VRAM
    embedding_batch_size: int = 16

    # bge-m3 outputs 1024-dimensional vectors
    # This must match what you tell Qdrant when creating the collection
    embedding_dim: int = 1024

    # ── Reranker ─────────────────────────────────────────────
    # Cross-encoder reranker — much more accurate than bi-encoder
    # but slower, so we only run it on top-K candidates
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Qdrant ───────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Collection = like a table in a database
    # All our medical chunks live in this one collection
    qdrant_collection: str = "medrag_docs"

    # ── Chunking ─────────────────────────────────────────────
    # chunk_size is measured in approximate tokens (1 token ≈ 4 chars)
    # 512 tokens = ~400 words — good balance for medical paragraphs
    chunk_size: int = 512

    # Overlap ensures context isn't lost at chunk boundaries
    # If a sentence splits across two chunks, both chunks contain it
    chunk_overlap: int = 64

    # Discard any chunk shorter than this — it's probably a header or noise
    min_chunk_size: int = 100

    # ── Retrieval ─────────────────────────────────────────────
    # How many candidates to fetch from dense (vector) search
    # We fetch 20, then rerank down to rerank_top_k=5
    # More candidates = better recall, slower reranking
    dense_top_k: int = 20

    # Same but for BM25 sparse search
    sparse_top_k: int = 20

    # After RRF fusion + reranking, keep only these top results
    # These 5 chunks go into the LLM context window
    rerank_top_k: int = 5

    # RRF (Reciprocal Rank Fusion) constant
    # Controls how much top ranks are boosted vs lower ranks
    # 60 is the standard value from the original RRF paper
    rrf_k: int = 60

    # ── PDF Parsing ──────────────────────────────────────────
    extract_tables: bool = True
    extract_images: bool = True

    # Images smaller than this (in pixels) are ignored
    # Filters out logos, decorative lines, small icons
    min_image_size: int = 100

    # Camelot table extraction mode:
    # "lattice" = tables with visible borders (most medical papers)
    # "stream"  = tables without borders (use as fallback)
    table_flavor: str = "lattice"

    # ── FastAPI ──────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Evaluation ───────────────────────────────────────────
    # Number of synthetic QA pairs to generate for RAGAS evaluation
    eval_sample_size: int = 50

    class Config:
        # Tell pydantic-settings where to find the .env file
        env_file = ".env"
        env_file_encoding = "utf-8"

        # If a variable is in .env but not defined above, ignore it
        # Without this, extra .env vars would raise an error
        extra = "ignore"


# ── Singleton ─────────────────────────────────────────────────
# We instantiate once here, at import time.
# Every file does: from config import settings
# They all get the same object — no repeated .env file reading
settings = Settings()


# ── Auto-create directories ───────────────────────────────────
# When config.py is first imported, ensure all data folders exist.
# This way no other file needs to worry about mkdir.
for _dir in [
    settings.data_dir,
    settings.pdf_dir,
    settings.index_dir,
    settings.cache_dir,
]:
    _dir.mkdir(parents=True, exist_ok=True)
    # parents=True  → creates intermediate dirs (like mkdir -p)
    # exist_ok=True → no error if dir already exists