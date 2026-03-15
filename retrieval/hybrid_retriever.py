# retrieval/hybrid_retriever.py
# -----------------------------
# Hybrid retrieval combining dense (Qdrant) + sparse (BM25)
# using Reciprocal Rank Fusion (RRF).
#
# Pipeline:
#   1. Embed query with bge-m3             (dense query vector)
#   2. Dense search in Qdrant             (top-K by cosine similarity)
#   3. BM25 search in local index         (top-K by keyword score)
#   4. RRF fusion                         (merge + rerank both lists)
#   5. Return top-K fused results
#
# The reranker (Sprint 3) takes these results and does a final
# more expensive reranking pass with a CrossEncoder.

from __future__ import annotations

import logging
from typing import Optional

from config import settings
from ingestion.embedder import get_embedder
from retrieval.bm25_index import BM25Index
from retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Retrieves relevant chunks using hybrid dense + sparse search.

    Dense search  → semantic similarity via Qdrant + bge-m3
    Sparse search → keyword matching via BM25
    RRF fusion    → combines both ranked lists into one

    Usage:
        retriever = HybridRetriever()
        results = retriever.retrieve("what is the dosage of metformin?")
        # Returns top-K chunks ranked by RRF score
    """

    def __init__(self):
        self.embedder     = get_embedder()
        self.vector_store = QdrantStore()
        self.bm25_index   = BM25Index()

        self.dense_top_k  = settings.dense_top_k   # 20 candidates from dense
        self.sparse_top_k = settings.sparse_top_k  # 20 candidates from BM25
        self.rrf_k        = settings.rrf_k          # 60, RRF constant
        self.final_top_k  = settings.rerank_top_k   # 5 after fusion

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = None,
        source_file: str = None,
    ) -> list[dict]:
        """
        Main retrieval method. Runs hybrid search and returns
        RRF-fused results.

        Args:
            query:       The user's question in plain text
            k:           How many results to return after fusion
                         Defaults to settings.rerank_top_k (5)
            source_file: Optional — restrict search to one PDF file
                         e.g. source_file="lancet_study.pdf"

        Returns:
            List of chunk dicts sorted by RRF score descending.
            Each dict has all MedDocument fields + "rrf_score".

        Example:
            [
                {
                    "content": "Metformin 500mg twice daily...",
                    "page_number": 4,
                    "section_title": "Methods",
                    "source_file": "diabetes_study.pdf",
                    "rrf_score": 0.0325,
                    "dense_rank": 2,
                    "sparse_rank": 1,
                },
                ...
            ]
        """
        k = k or self.final_top_k

        logger.info(f"Hybrid retrieval: '{query[:80]}'")

        # ── Step 1: Dense retrieval ───────────────────────────
        # Embed the query and search Qdrant for nearest vectors
        query_vector = self.embedder.embed_query(query)
        # Uses QUERY_PREFIX internally — asymmetric retrieval ✅

        dense_results = self.vector_store.search(
            query_vector=query_vector,
            k=self.dense_top_k,
            source_file=source_file,
        )
        # Returns up to 20 chunks sorted by cosine similarity

        logger.info(f"  Dense:  {len(dense_results)} results")

        # ── Step 2: Sparse retrieval ──────────────────────────
        # BM25 keyword search — no embedding needed
        if self.bm25_index.is_ready():
            sparse_results = self.bm25_index.search(
                query=query,
                k=self.sparse_top_k,
            )
            logger.info(f"  Sparse: {len(sparse_results)} results")
        else:
            # BM25 not built yet — fall back to dense only
            logger.warning("BM25 index not ready — using dense search only")
            sparse_results = []

        # ── Step 3: RRF Fusion ────────────────────────────────
        fused = self._rrf_fusion(dense_results, sparse_results)
        logger.info(f"  Fused:  {len(fused)} unique results")

        # ── Step 4: Return top-K ──────────────────────────────
        final = fused[:k]
        logger.info(
            f"  Returning top {len(final)} "
            f"(top RRF score: {final[0]['rrf_score']:.4f})"
            if final else "  No results found"
        )

        return final

    # ── Private: RRF Fusion ───────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
    ) -> list[dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF formula for each chunk:
            rrf_score = Σ 1 / (k + rank)
            where the sum is over each list the chunk appears in.

        A chunk appearing in both lists scores higher than one
        appearing in only one list — even if its rank in that
        single list is very high.

        Args:
            dense_results:  Chunks from Qdrant (ranked by cosine similarity)
            sparse_results: Chunks from BM25 (ranked by BM25 score)

        Returns:
            Combined list sorted by RRF score descending.
            Each result has added fields:
              rrf_score, dense_rank, sparse_rank
        """
        # ── Build score accumulator ───────────────────────────
        # Key: chunk identity string
        # Value: dict accumulating RRF score + metadata
        rrf_scores: dict[str, dict] = {}

        # ── Process dense results ─────────────────────────────
        for rank, result in enumerate(dense_results):
            # rank is 0-indexed — RRF uses 1-indexed ranks
            # so we add 1: rank=0 → position 1

            chunk_key = self._make_key(result)
            # Unique string identifying this chunk —
            # same chunk from both lists must produce same key

            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            # Core RRF formula:
            # rank=0 (best) → 1/(60+1) = 0.01639
            # rank=1        → 1/(60+2) = 0.01613
            # rank=19 (worst) → 1/(60+20) = 0.01250

            if chunk_key not in rrf_scores:
                # First time seeing this chunk — initialize its entry
                rrf_scores[chunk_key] = {
                    **result,
                    # Copy all chunk fields (content, page, section, etc.)
                    "rrf_score": 0.0,    # will accumulate
                    "dense_rank": rank + 1,
                    "sparse_rank": None,
                    # None means this chunk wasn't found by BM25
                }

            rrf_scores[chunk_key]["rrf_score"] += rrf_score
            # += because the same chunk might appear in both lists —
            # we accumulate scores from each list

        # ── Process sparse results ────────────────────────────
        for rank, result in enumerate(sparse_results):
            chunk_key = self._make_key(result)
            rrf_score = 1.0 / (self.rrf_k + rank + 1)

            if chunk_key not in rrf_scores:
                # Chunk only in BM25, not in dense results
                rrf_scores[chunk_key] = {
                    **result,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    # None means dense search didn't find this chunk
                    "sparse_rank": rank + 1,
                }
            else:
                # Chunk appeared in BOTH lists — add BM25 score to existing
                rrf_scores[chunk_key]["rrf_score"] += rrf_score
                rrf_scores[chunk_key]["sparse_rank"] = rank + 1

        # ── Sort by RRF score descending ──────────────────────
        fused = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
            # reverse=True → highest score first
        )

        # ── Log fusion stats ──────────────────────────────────
        both_lists = sum(
            1 for r in fused
            if r["dense_rank"] is not None and r["sparse_rank"] is not None
        )
        logger.info(
            f"  RRF: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(fused)} unique ({both_lists} in both lists)"
        )

        return fused

    def _make_key(self, result: dict) -> str:
        """
        Create a unique string key for a chunk result.

        Used to identify the same chunk appearing in both
        dense and sparse result lists.

        We use source_file + page_number + chunk_index
        because these three fields uniquely identify a chunk —
        the same combination used in QdrantStore._make_id().
        """
        return (
            f"{result.get('source_file', '')}__"
            f"{result.get('page_number', '')}__"
            f"{result.get('chunk_index', '')}"
        )