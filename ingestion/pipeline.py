"""
pipeline.py
-----------
Master ingestion pipeline.
Orchestrates: Parse → Caption → Chunk → Embed → Index (Qdrant + BM25)

Usage:
    from ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    pipeline.ingest("data/raw/paper.pdf")
"""

import logging
import pickle
from pathlib import Path
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from rank_bm25 import BM25Okapi

from ingestion.pdf_parser import PDFParser
from ingestion.image_captioner import ImageCaptioner
from ingestion.chunker import MedicalChunker
from ingestion.embedder import BGEEmbedder
from config import settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline for medical PDFs.

    Steps:
    1. PDFParser     → RawPage objects (text + tables + images)
    2. ImageCaptioner → fills RawImage.caption via LLaVA-Phi3
    3. MedicalChunker → LangChain Documents with metadata
    4. BGEEmbedder   → dense float vectors
    5. Qdrant        → store dense vectors + metadata
    6. BM25          → build sparse index for hybrid retrieval
    """

    def __init__(self, skip_image_captioning: bool = False):
        logger.info("Initializing ingestion pipeline...")
        self.parser = PDFParser(extract_images=True, extract_tables=True)
        self.captioner = ImageCaptioner() if not skip_image_captioning else None
        self.chunker = MedicalChunker()
        self.embedder = BGEEmbedder()
        self.qdrant = self._init_qdrant()
        self.bm25_index: BM25Okapi | None = None
        self.bm25_docs: list[Document] = []

        # Ensure index dir exists
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        settings.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, pdf_path: str | Path) -> dict:
        """
        Full ingestion pipeline for one PDF.
        Returns a summary dict with stats.
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Starting ingestion: {pdf_path.name}")

        # Step 1: Parse PDF
        pages = self.parser.parse(pdf_path)

        # Step 2: Caption images (skipped if captioner is None)
        if self.captioner:
            pages = self.captioner.caption_all(pages)

        # Step 3: Chunk into Documents
        docs = self.chunker.chunk(pages)

        # Step 4: Generate embeddings
        embeddings = self.embedder.embed_documents(docs)

        # Step 5: Upsert into Qdrant
        self._upsert_to_qdrant(docs, embeddings)

        # Step 6: Update BM25 index
        self._update_bm25(docs)

        stats = {
            "file": pdf_path.name,
            "pages": len(pages),
            "documents": len(docs),
            "text_chunks": sum(1 for d in docs if d.metadata["content_type"] == "text"),
            "table_chunks": sum(1 for d in docs if d.metadata["content_type"] == "table"),
            "figure_chunks": sum(1 for d in docs if d.metadata["content_type"] == "figure"),
        }
        logger.info(f"Ingestion complete: {stats}")
        return stats

    def ingest_directory(self, dir_path: str | Path) -> list[dict]:
        """Ingest all PDFs in a directory."""
        dir_path = Path(dir_path)
        pdfs = list(dir_path.glob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDFs in {dir_path}")
        return [self.ingest(pdf) for pdf in pdfs]

    # ── Qdrant ────────────────────────────────────────────────────────────────

    def _init_qdrant(self) -> QdrantClient:
        """Connect to Qdrant and create collection if needed."""
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        existing = [c.name for c in client.get_collections().collections]
        if settings.qdrant_collection not in existing:
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")
        else:
            logger.info(f"Using existing Qdrant collection: {settings.qdrant_collection}")

        return client

    def _upsert_to_qdrant(self, docs: list[Document], embeddings: list[list[float]]):
        """Upsert documents and their embeddings into Qdrant."""
        points = []
        for i, (doc, vector) in enumerate(zip(docs, embeddings)):
            # Qdrant point ID: hash of chunk_id for idempotent upserts
            point_id = abs(hash(doc.metadata["chunk_id"])) % (2**63)

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "page_content": doc.page_content,
                    **doc.metadata,
                },
            ))

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant.upsert(
                collection_name=settings.qdrant_collection,
                points=batch,
            )
        logger.info(f"Upserted {len(points)} points to Qdrant")

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _update_bm25(self, new_docs: list[Document]):
        """
        Add new documents to the BM25 index.
        Persists index to disk for reuse across runs.
        """
        index_path = settings.index_dir / "bm25.pkl"
        docs_path = settings.index_dir / "bm25_docs.pkl"

        # Load existing if available
        if index_path.exists() and docs_path.exists():
            with open(docs_path, "rb") as f:
                self.bm25_docs = pickle.load(f)
            logger.info(f"Loaded existing BM25 index ({len(self.bm25_docs)} docs)")

        # Merge new docs
        self.bm25_docs.extend(new_docs)

        # Rebuild index (BM25Okapi requires full rebuild on update)
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.bm25_docs]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # Persist
        with open(index_path, "wb") as f:
            pickle.dump(self.bm25_index, f)
        with open(docs_path, "wb") as f:
            pickle.dump(self.bm25_docs, f)

        logger.info(f"BM25 index updated and saved ({len(self.bm25_docs)} total docs)")
