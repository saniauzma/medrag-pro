# ingestion/pipeline.py
# ---------------------
# Orchestrates the full ingestion pipeline:
#   PDF → Parse → Chunk → Embed → Store (Qdrant + BM25)
#
# This is the single entry point for ingesting medical PDFs.
# All other ingestion modules (parser, chunker, embedder) are
# called from here in sequence.
#
# Usage:
#   pipeline = IngestionPipeline()
#   result = pipeline.run("data/pdfs/")           # whole directory
#   result = pipeline.run("data/pdfs/paper.pdf")  # single file

from __future__ import annotations

import logging
import time
# time.time() lets us measure how long ingestion takes.
# Useful for profiling — if ingestion is slow, you know which step to optimize.

from pathlib import Path

from config import settings
from ingestion.pdf_parser import MedPDFParser, MedDocument
from ingestion.chunker import MedChunker
from ingestion.embedder import MedEmbedder
from ingestion.embedder import get_embedder


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Full ingestion pipeline: PDF → Qdrant + BM25.

    Wires together the parser, chunker, and embedder.
    Vector store and BM25 index are lazy-loaded from the
    retrieval module (written in Sprint 2).

    Design principle: each step is independent and testable.
    You can call parser, chunker, embedder separately in tests
    without running the full pipeline.
    """

    def __init__(self):
        # These three are always needed — initialize eagerly
        self.parser = MedPDFParser()
        self.chunker = MedChunker()
        self.embedder = get_embedder()

        # These come from Sprint 2 — lazy load to avoid import errors now
        self._vector_store = None
        self._bm25_index = None

    # ── Lazy properties ───────────────────────────────────────────────────────

    @property
    def vector_store(self):
        """
        Lazy-load QdrantStore from retrieval module.

        @property means you access it like an attribute: self.vector_store
        but it runs this function the first time you access it.

        First access:   imports QdrantStore, creates instance, caches it
        Later accesses: returns the cached instance (self._vector_store)

        This pattern is called "lazy initialization" — we defer the
        expensive work (connecting to Qdrant) until it's actually needed.
        """
        if self._vector_store is None:
            from retrieval.vector_store import QdrantStore
            # This import only runs once — when vector_store is first accessed.
            # If retrieval/vector_store.py doesn't exist yet, only this line
            # fails — not the entire pipeline module at import time.
            self._vector_store = QdrantStore()
        return self._vector_store

    @property
    def bm25_index(self):
        """Lazy-load BM25Index from retrieval module."""
        if self._bm25_index is None:
            from retrieval.bm25_index import BM25Index
            self._bm25_index = BM25Index()
        return self._bm25_index

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, path: str | Path) -> dict:
        """
        Run the full ingestion pipeline on a file or directory.

        Returns a summary dict — useful for logging, API responses,
        and debugging. Never returns None so callers can always
        safely access result["status"].

        Args:
            path: Path to a single PDF file or a directory of PDFs.

        Returns:
            {
                "status": "success" | "empty" | "error",
                "total_chunks": 243,
                "text_chunks": 198,
                "table_chunks": 31,
                "image_chunks": 14,
                "avg_tokens": 487.2,
                "elapsed_seconds": 42.3
            }
        """
        path = Path(path)
        start_time = time.time()
        # Record start time — we'll subtract at the end to get elapsed

        # ── Step 1: Parse ─────────────────────────────────────
        logger.info("=" * 50)
        logger.info("STEP 1/4: Parsing PDFs")
        logger.info("=" * 50)

        if path.is_dir():
            raw_docs = self.parser.parse_directory(path)
        elif path.is_file():
            raw_docs = self.parser.parse(path)
        else:
            # Path doesn't exist at all
            raise FileNotFoundError(f"Path not found: {path}")

        if not raw_docs:
            # Parser ran but produced nothing — warn and exit early.
            # This happens if the PDF is empty, image-only, or corrupted.
            logger.warning("No documents parsed. Check your PDF files.")
            return {"status": "empty", "total_chunks": 0, "elapsed_seconds": 0}

        logger.info(f"Parsed {len(raw_docs)} raw documents")

        # ── Step 2: Chunk ─────────────────────────────────────
        logger.info("=" * 50)
        logger.info("STEP 2/4: Chunking documents")
        logger.info("=" * 50)

        chunks = self.chunker.chunk(raw_docs)
        stats = self.chunker.get_stats(chunks)
        # stats = {"total_chunks": 243, "text_chunks": 198, ...}

        logger.info(f"Chunk stats: {stats}")

        if not chunks:
            logger.warning("Chunking produced no output.")
            return {"status": "empty", "total_chunks": 0, "elapsed_seconds": 0}

        # ── Step 3: Embed ─────────────────────────────────────
        logger.info("=" * 50)
        logger.info("STEP 3/4: Embedding chunks")
        logger.info("=" * 50)

        embeddings = self.embedder.embed_documents(chunks)
        # embeddings is a list of 1024-dim vectors.
        # len(embeddings) == len(chunks) — guaranteed by embed_documents.

        logger.info(f"Generated {len(embeddings)} embeddings "
                    f"(dim={len(embeddings[0]) if embeddings else 0})")

        # ── Step 4: Store ─────────────────────────────────────
        logger.info("=" * 50)
        logger.info("STEP 4/4: Storing in Qdrant + BM25")
        logger.info("=" * 50)

        # Store dense vectors in Qdrant
        self.vector_store.upsert(chunks, embeddings)
        # upsert = insert if not exists, update if exists.
        # Safe to call multiple times on the same PDF — no duplicates.

        # Build sparse BM25 index and save to disk
        self.bm25_index.build(chunks)
        # BM25 index is rebuilt from scratch each time.
        # For large document sets we'd do incremental updates —
        # but rebuild is simpler and correct for now.

        # ── Result ────────────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)

        result = {
            "status": "success",
            "elapsed_seconds": elapsed,
            **stats,
            # ** unpacks stats dict into result dict.
            # Equivalent to manually adding each stats key.
        }

        logger.info("=" * 50)
        logger.info(f"Ingestion complete: {result}")
        logger.info("=" * 50)

        return result

    def run_partial(self, path: str | Path) -> list[MedDocument]:
        """
        Run only the parse + chunk steps — no embedding or storing.

        Useful for:
          - Testing the parser and chunker without needing Qdrant running
          - Inspecting what chunks look like before committing to the index
          - Unit testing individual pipeline stages

        Usage:
            chunks = pipeline.run_partial("data/pdfs/paper.pdf")
            for chunk in chunks[:5]:
                print(chunk.content_type, chunk.content[:100])
        """
        path = Path(path)

        raw_docs = (
            self.parser.parse_directory(path)
            if path.is_dir()
            else self.parser.parse(path)
        )
        # Ternary expression:
        # condition_is_true if condition else condition_is_false
        # Cleaner than a full if/else for simple assignments.

        chunks = self.chunker.chunk(raw_docs)
        logger.info(f"Partial run: {len(chunks)} chunks ready (not stored)")
        return chunks


# ─── CLI ──────────────────────────────────────────────────────────────────────
# Run this file directly to ingest PDFs from the command line:
#   python -m ingestion.pipeline                         ← uses default pdf_dir
#   python -m ingestion.pipeline data/pdfs/paper.pdf    ← single file
#   python -m ingestion.pipeline data/pdfs/             ← whole directory

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        # Output looks like: 14:23:01 | INFO | STEP 1/4: Parsing PDFs
    )

    # Use command line argument if provided, else fall back to config default
    target_path = sys.argv[1] if len(sys.argv) > 1 else str(settings.pdf_dir)

    pipeline = IngestionPipeline()
    result = pipeline.run(target_path)

    print(f"\nResult: {result}")