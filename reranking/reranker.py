# reranking/reranker.py
# ---------------------
# CrossEncoder reranker for final precision pass.
#
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2
#   - Trained on MS MARCO passage ranking dataset
#   - 6-layer MiniLM — fast enough for real-time reranking
#   - Outputs a single relevance score per (query, passage) pair
#   - Runs on CPU — small enough that GPU isn't needed
#   - ~80MB — downloads once, cached locally
#
# Why CPU for reranker but GPU for embeddings?
#   Embeddings: process thousands of chunks in parallel → GPU batch wins
#   Reranker:   process 10 pairs sequentially → GPU overhead not worth it
#   On 4GB VRAM this also avoids competing with bge-m3 for GPU memory

from __future__ import annotations

import logging
from typing import Optional

from sentence_transformers import CrossEncoder
# CrossEncoder is different from SentenceTransformer:
#   SentenceTransformer: encodes single texts → vectors
#   CrossEncoder: scores pairs of texts → scalar relevance score

from config import settings

logger = logging.getLogger(__name__)


class MedReranker:
    """
    Reranks retrieved chunks using a CrossEncoder model.

    Takes the top-K candidates from hybrid retrieval and
    produces a more accurate final ranking by scoring each
    (query, chunk) pair together.

    Usage:
        reranker = MedReranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self):
        logger.info(f"Loading reranker: {settings.reranker_model}")

        self.model = CrossEncoder(
            settings.reranker_model,
            # "cross-encoder/ms-marco-MiniLM-L-6-v2"
            # Downloads ~80MB on first run, cached after that.

            device="cpu",
            # Explicitly CPU — see module docstring for why.
            # Even with CUDA available, CPU is better here:
            #   - Avoids competing with bge-m3 for 4GB VRAM
            #   - MiniLM is small — CPU inference is fast enough
            #   - 10 pairs takes ~200ms on CPU — acceptable latency
        )

        logger.info("Reranker ready ✅")

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = None,
    ) -> list[dict]:
        """
        Rerank a list of candidate chunks for a given query.

        Args:
            query:      The user's original question
            candidates: List of chunk dicts from HybridRetriever.retrieve()
                        Each dict must have a "content" field.
            top_k:      How many reranked results to return.
                        Defaults to settings.rerank_top_k (5)

        Returns:
            Top-k chunks sorted by CrossEncoder relevance score.
            Each chunk gets a new "rerank_score" field added.
            Original RRF scores are preserved as "rrf_score".

        Example:
            candidates = retriever.retrieve("metformin dosage", k=10)
            reranked   = reranker.rerank("metformin dosage", candidates, top_k=5)
            # reranked[0] has highest CrossEncoder relevance score
        """
        top_k = top_k or settings.rerank_top_k

        if not candidates:
            logger.warning("No candidates to rerank")
            return []

        if len(candidates) == 1:
            # Nothing to rerank with a single candidate
            candidates[0]["rerank_score"] = 1.0
            return candidates

        logger.info(f"Reranking {len(candidates)} candidates...")

        # ── Build (query, passage) pairs ──────────────────────
        pairs = [
            [query, candidate["content"]]
            for candidate in candidates
        ]
        # CrossEncoder expects a list of [query, passage] pairs.
        # It processes all pairs in one forward pass — efficient.
        #
        # Example pairs:
        # [
        #   ["what is metformin dosage?", "Metformin 500mg twice daily..."],
        #   ["what is metformin dosage?", "Clinical trials showed that..."],
        #   ...
        # ]

        # ── Score all pairs ───────────────────────────────────
        scores = self.model.predict(
            pairs,
            # Returns a numpy array of shape (num_candidates,)
            # scores[i] = relevance of candidates[i] to query
            # Higher = more relevant

            show_progress_bar=False,
            # No progress bar for reranking — it's fast (10 pairs)
            # Progress bars make sense for embedding 1000+ chunks,
            # not for scoring 10 pairs.
        )
        # scores is a raw logit — can be any float, not bounded 0-1.
        # ms-marco CrossEncoders output logits, not probabilities.
        # We don't need to normalize — relative ordering is what matters.

        # ── Attach scores to candidates ───────────────────────
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
            # float() converts numpy float32 → Python float
            # Important for JSON serialization (numpy types aren't JSON-serializable)

        # ── Sort by rerank score descending ───────────────────
        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True,
            # reverse=True → highest score first (most relevant)
        )

        # ── Return top-k ──────────────────────────────────────
        final = reranked[:top_k]

        logger.info(
            f"Reranking complete: "
            f"top score={final[0]['rerank_score']:.3f} "
            f"bottom score={final[-1]['rerank_score']:.3f}"
        )

        # ── Log rank changes ──────────────────────────────────
        # This is useful for understanding how much reranking changes things.
        # If reranking always produces the same order as RRF, it's not adding value.
        self._log_rank_changes(candidates, reranked, top_k)

        return final

    def rerank_with_scores(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[tuple[dict, float]]:
        """
        Rerank and return (chunk, score) tuples — all candidates, not just top-k.

        Useful for evaluation — you can inspect the full score distribution
        to understand how the reranker separates relevant from irrelevant chunks.

        Usage:
            scored = reranker.rerank_with_scores(query, candidates)
            for chunk, score in scored:
                print(f"{score:.3f} | {chunk['content'][:80]}")
        """
        if not candidates:
            return []

        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        scored = list(zip(candidates, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── Private ───────────────────────────────────────────────────────────────

    def _log_rank_changes(
        self,
        original: list[dict],
        reranked: list[dict],
        top_k: int,
    ) -> None:
        """
        Log how much the reranker changed the ranking vs RRF.

        This is a development-time diagnostic. In production you'd
        track this as a metric to monitor reranker effectiveness.

        Example output:
            Rank changes (RRF → Reranker):
              #1 stayed #1  | score=8.234 | "Metformin 500mg twice..."
              #4 → #2       | score=7.891 | "Standard dosing protocol..."
              #2 → #3       | score=6.543 | "Clinical trials showed..."
        """
        # Build original rank lookup: content → original position
        original_ranks = {
            c["content"][:50]: i + 1
            for i, c in enumerate(original)
        }
        # We use first 50 chars of content as key —
        # unique enough to identify chunks, short enough to not waste memory

        logger.info("Rank changes (RRF → Reranker):")
        for new_rank, chunk in enumerate(reranked[:top_k], start=1):
            key = chunk["content"][:50]
            old_rank = original_ranks.get(key, "?")

            if old_rank == new_rank:
                change = f"stayed #{new_rank}"
            else:
                change = f"#{old_rank} → #{new_rank}"

            logger.info(
                f"  {change:15} | "
                f"score={chunk['rerank_score']:6.3f} | "
                f"\"{chunk['content'][:60]}...\""
            )