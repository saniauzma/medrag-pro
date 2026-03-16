# retrieval/vector_store.py
# -------------------------
# Qdrant vector store wrapper.
#
# Responsibilities:
#   1. Create and configure the Qdrant collection on first run
#   2. Upsert chunks + embeddings during ingestion
#   3. Dense vector search at query time
#
# Qdrant runs locally in Docker on port 6333.
# No API key needed — fully local and free.

from __future__ import annotations

import hashlib
# We use hashlib to generate deterministic IDs for each chunk.
# ID = MD5 hash of (source_file + page_number + chunk_index).
# This means the same chunk always gets the same ID —
# which is what makes upsert idempotent (no duplicates on re-ingestion).

import logging
import uuid
# uuid is the standard format Qdrant expects for point IDs.
# We convert our MD5 hash into a UUID so Qdrant accepts it.

from qdrant_client import QdrantClient
# The official Qdrant Python client.
# Handles HTTP communication with the Qdrant server.

from qdrant_client.models import (
    Distance,
    # Enum for distance metric:
    # Distance.COSINE  → cosine similarity (angle between vectors)
    # Distance.DOT     → dot product (fast when vectors are normalized)
    # Distance.EUCLID  → euclidean distance (L2 norm)
    # We use DOT because our embeddings are L2-normalized —
    # dot product == cosine similarity for normalized vectors, but faster.

    PointStruct,
    # Represents a single point to upsert into Qdrant.
    # PointStruct(id=..., vector=..., payload=...)

    VectorParams,
    # Configuration for the vector space:
    # VectorParams(size=1024, distance=Distance.DOT)
    # Must match the embedding model's output dimension exactly.

    Filter,
    FieldCondition,
    MatchValue,
    # Used for metadata pre-filtering during search.
    # e.g. only search chunks from a specific PDF file.
)

from config import settings
from ingestion.pdf_parser import MedDocument
import hashlib
import uuid

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Manages the Qdrant vector collection for MedRAG Pro.

    Handles:
      - Collection creation with correct vector config
      - Upserting chunks + embeddings (idempotent)
      - Dense vector similarity search
      - Optional metadata pre-filtering

    Usage:
        store = QdrantStore()
        store.upsert(chunks, embeddings)           # during ingestion
        results = store.search(query_vector, k=20) # during retrieval
    """

    def __init__(self):
        # Connect to local Qdrant instance running in Docker
        self.client = QdrantClient(
            host=settings.qdrant_host,   # "localhost"
            port=settings.qdrant_port,   # 6333
            timeout=30
        )
        self.collection_name = settings.qdrant_collection
        # "medrag_docs" — all our chunks live in this one collection

        # Ensure collection exists with correct config
        self._ensure_collection()

    # ── Collection Setup ──────────────────────────────────────────────────────

    def _ensure_collection(self):
        """
        Create the Qdrant collection if it doesn't exist.
        Safe to call on every startup — does nothing if already exists.

        Why check first instead of always creating?
        Calling create_collection on an existing collection raises an error.
        We want startup to be idempotent — run the app 100 times, same result.
        """
        # Get list of existing collections
        existing = [
            c.name
            for c in self.client.get_collections().collections
        ]
        # .get_collections() returns a CollectionsResponse object.
        # .collections is a list of CollectionDescription objects.
        # We extract just the names into a plain list.

        if self.collection_name not in existing:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    # 1024 — must match BAAI/bge-m3 output dimension exactly.
                    # If this doesn't match the actual embedding size,
                    # Qdrant will reject every upsert with a dimension error.

                    distance=Distance.DOT,
                    # Dot product distance.
                    # Works as cosine similarity because our embeddings
                    # are L2-normalized (normalize_embeddings=True in embedder).
                    # Dot product is slightly faster than cosine in Qdrant's HNSW index.
                ),
            )
            logger.info(f"Collection '{self.collection_name}' created ✅")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists ✅")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def upsert(
        self,
        chunks: list[MedDocument],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store chunks and their embeddings in Qdrant.

        chunks and embeddings must be the same length and in the same order —
        chunks[i] corresponds to embeddings[i].

        Uses batched upsert for efficiency:
          Sending 1000 points one by one = 1000 HTTP requests
          Sending 1000 points in batches of 100 = 10 HTTP requests
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must have the same length"
            )
            # Fail loudly — a mismatch here means wrong data gets stored,
            # which would silently corrupt retrieval results.

        if not chunks:
            logger.warning("No chunks to upsert")
            return

        # Build PointStruct objects — one per chunk
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # zip() pairs up chunks[i] with embeddings[i] cleanly.
            # Much safer than indexing with chunks[i], embeddings[i].

            point_id = self._make_id(chunk)
            # Deterministic UUID based on chunk identity.
            # Same chunk → same ID → upsert is idempotent.

            payload = chunk.to_dict()
            # Flatten the MedDocument into a plain dict.
            # Qdrant stores this as metadata alongside the vector.
            # We can filter and retrieve by any payload field at query time.

            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            ))

        # Batch upsert — send in chunks of 100 to avoid large HTTP payloads
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        # Ceiling division: 243 points / 100 = 3 batches (not 2.43)
        # Formula: (n + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start = batch_num * batch_size
            end = start + batch_size
            batch = points[start:end]
            # Slice the points list into batches.
            # Last batch may be smaller than batch_size — that's fine.

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.info(
                f"  Upserted batch {batch_num + 1}/{total_batches} "
                f"({len(batch)} points)"
            )

        logger.info(f"Upsert complete: {len(points)} points in Qdrant ✅")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        k: int = None,
        source_file: str = None,
    ) -> list[dict]:
        """
        Dense vector similarity search.
        Uses query_points() — required for qdrant-client >= 1.9
        """
        k = k or settings.dense_top_k

        # ── Optional metadata pre-filter ──────────────────────
        query_filter = None
        if source_file:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    )
                ]
            )

        # ── Execute search ────────────────────────────────────
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            # New API: "query" instead of "query_vector"

            limit=k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        ).points
        # .points extracts the list from the QueryResponse wrapper object

        # ── Format results ────────────────────────────────────
        formatted = []
        for result in results:
            doc = result.payload
            doc["score"] = result.score
            formatted.append(doc)

        logger.debug(
            f"Dense search returned {len(formatted)} results "
            f"(top score: {formatted[0]['score']:.3f})" if formatted else
            "Dense search returned 0 results"
        )

        return formatted
    
    def _make_id(self, chunk: MedDocument) -> str:
        """
        Generate a deterministic UUID for a chunk.
        Same chunk always gets the same ID — makes upsert idempotent.
        """
        key = f"{chunk.source_file}_{chunk.page_number}_{chunk.chunk_index}"
        md5_hash = hashlib.md5(key.encode()).hexdigest()
        return str(uuid.UUID(md5_hash))

    def delete_collection(self) -> None:
        """Delete and recreate the collection — use during development to reset."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Collection '{self.collection_name}' deleted ⚠️")
        self._ensure_collection()

    
    def get_collection_info(self) -> dict:
        """Returns info about the current collection — useful for health checks."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "total_points": info.points_count,
            "status": info.status,
        "vector_size": info.config.params.vectors.size,
        "distance": info.config.params.vectors.distance,
    }