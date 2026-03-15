# retrieval/bm25_index.py
# -----------------------
# BM25 sparse retrieval index for keyword-based search.
#
# BM25 complements dense vector search:
#   Dense  → finds semantically similar chunks (meaning-based)
#   BM25   → finds exact keyword matches (term-based)
#   RRF    → combines both for best overall recall
#
# The index is built from all chunks at ingestion time
# and saved to disk so it persists across restarts.

from __future__ import annotations

import logging
import pickle
# pickle serializes Python objects to bytes for saving to disk.
# We use it to save/load the BM25 index and the chunk corpus.
# Note: pickle is safe here because we're only loading our own files.

import re
# For simple tokenization — splitting text into words.

from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi
# BM25Okapi is the standard BM25 variant.
# "Okapi" refers to the Okapi BM25 formula from the original paper.
# It handles term frequency saturation and document length normalization.

from config import settings
from ingestion.pdf_parser import MedDocument

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Sparse keyword retrieval using BM25Okapi.

    Built from all ingested chunks at ingestion time.
    Saved to disk and loaded on subsequent runs.

    Usage:
        index = BM25Index()
        index.build(chunks)                    # during ingestion
        results = index.search("metformin dosage", k=20)  # during retrieval
    """

    # File paths for persisting the index to disk
    # We save two files:
    #   bm25.pkl   → the BM25Okapi object (contains term statistics)
    #   corpus.pkl → the original MedDocument chunks (for retrieving content)
    INDEX_FILE  = "bm25.pkl"
    CORPUS_FILE = "bm25_corpus.pkl"

    def __init__(self):
        self.index_dir  = settings.index_dir
        # "data/indexes/" — where we save the index files

        self.index_path  = self.index_dir / self.INDEX_FILE
        self.corpus_path = self.index_dir / self.CORPUS_FILE

        self.bm25:   Optional[BM25Okapi]     = None
        self.corpus: Optional[list[MedDocument]] = None
        # Both start as None — populated by build() or load()

        # Try to load existing index from disk on startup
        if self.index_path.exists() and self.corpus_path.exists():
            self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, chunks: list[MedDocument]) -> None:
        """
        Build the BM25 index from a list of MedDocument chunks.
        Saves the index to disk immediately after building.

        Called during ingestion after all chunks are created.
        Rebuilds from scratch each time — simple and correct.

        Args:
            chunks: All MedDocument chunks to index.
                    Should include TEXT, TABLE, and IMAGE chunks.
        """
        if not chunks:
            logger.warning("No chunks provided to BM25 index")
            return

        logger.info(f"Building BM25 index from {len(chunks)} chunks...")

        # Store the full corpus — needed to retrieve content at query time
        self.corpus = chunks

        # Tokenize each chunk for BM25
        tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in chunks
        ]
        # tokenized_corpus is a list of lists:
        # [["metformin", "reduces", "glucose"], ["clinical", "trial", ...], ...]
        # BM25Okapi expects exactly this format.

        # Build the index
        self.bm25 = BM25Okapi(tokenized_corpus)
        # BM25Okapi computes term frequencies and IDF scores here.
        # This is the "training" step — happens once at ingestion time.

        logger.info("BM25 index built ✅")

        # Save to disk immediately
        self._save()

    def search(
        self,
        query: str,
        k: int = None,
    ) -> list[dict]:
        """
        Search the BM25 index for chunks matching the query.

        Args:
            query: The raw user question (not tokenized — we handle that here)
            k:     Number of results to return

        Returns:
            List of dicts with chunk content + BM25 score.
            Sorted by score descending (most relevant first).

        Example:
            results = index.search("metformin dosage type 2 diabetes", k=20)
            # Returns chunks with highest BM25 score for these keywords
        """
        if self.bm25 is None or self.corpus is None:
            logger.warning("BM25 index not built yet. Call build() first.")
            return []

        k = k or settings.sparse_top_k
        # Default to config value (20) if not specified

        # Tokenize the query the same way we tokenized documents
        # CRITICAL: must use same tokenization for query and documents.
        # If docs are lowercased but query isn't, "Metformin" won't match "metformin".
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            logger.warning(f"Query tokenized to empty list: '{query}'")
            return []

        # Get BM25 scores for ALL documents in the corpus
        scores = self.bm25.get_scores(tokenized_query)
        # Returns a numpy array of shape (num_chunks,)
        # scores[i] = BM25 score of corpus[i] for this query
        # Higher = more relevant

        # Get indices of top-k scores
        # argsort returns indices sorted by value ascending
        # [-k:] takes the last k (highest scores)
        # [::-1] reverses to get descending order
        import numpy as np
        top_k_indices = np.argsort(scores)[-k:][::-1]
        # Example: if scores = [0.1, 0.8, 0.3, 0.9, 0.2]
        # argsort = [0, 4, 2, 1, 3]  (ascending)
        # [-3:]   = [2, 1, 3]         (top 3 indices)
        # [::-1]  = [3, 1, 2]         (descending)
        # scores[3]=0.9, scores[1]=0.8, scores[2]=0.3

        # Build result list — skip zero-score chunks
        results = []
        for idx in top_k_indices:
            score = float(scores[idx])
            # float() converts numpy float to Python float
            # Important for JSON serialization later

            if score <= 0.0:
                # BM25 score of 0 means no query terms appeared in this chunk.
                # No point including it — it's not a match at all.
                continue

            chunk = self.corpus[idx]
            doc = chunk.to_dict()
            doc["score"] = score
            doc["retrieval_source"] = "bm25"
            # Tag results with their source so hybrid_retriever knows
            # which came from BM25 vs dense search when doing RRF fusion.

            results.append(doc)

        logger.debug(
            f"BM25 search: '{query[:50]}' → {len(results)} results "
            f"(top score: {results[0]['score']:.3f})" if results else
            f"BM25 search: '{query[:50]}' → 0 results"
        )

        return results

    def is_ready(self) -> bool:
        """
        Check if the index is built and ready for search.
        Used by the API health check endpoint.
        """
        return self.bm25 is not None and self.corpus is not None

    def get_stats(self) -> dict:
        """
        Return stats about the current index.
        Useful for debugging and monitoring.
        """
        if not self.is_ready():
            return {"status": "not built"}

        return {
            "status": "ready",
            "total_docs": len(self.corpus),
            "vocab_size": len(self.bm25.idf),
            # idf is a dict of {term: idf_score}
            # vocab_size = number of unique terms in the corpus
            "index_file": str(self.index_path),
        }

    # ── Private: Tokenization ─────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into a list of lowercase words.

        Simple but effective for medical text:
          "Metformin (500mg) reduces HbA1c levels significantly."
          → ["metformin", "500mg", "reduces", "hba1c", "levels", "significantly"]

        Design decisions:
          1. Lowercase everything — case-insensitive matching
          2. Split on non-alphanumeric characters — handles punctuation
          3. Keep numbers — "500mg", "p<0.05", "HbA1c" are meaningful
          4. Filter short tokens — removes "a", "in", "of" noise
          5. No stemming/lemmatization — keeps medical terms intact
             "diabetic" and "diabetes" are different enough in clinical context

        Why not use NLTK or spaCy?
          They're heavy dependencies for a simple tokenization step.
          BM25 doesn't need linguistic sophistication — frequency statistics
          work well with simple word splitting.
        """
        # Lowercase
        text = text.lower()

        # Split on anything that isn't a letter, digit, or percent sign
        # re.split returns a list of tokens between matches
        tokens = re.split(r"[^a-z0-9%]+", text)
        # "metformin (500mg) reduces" → ["metformin", "500mg", "reduces"]

        # Filter: remove empty strings and very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        # Removes: "", "a", "in", "of", "to", "mg" (ambiguous)
        # Keeps: "metformin", "500mg", "reduces", "hba1c"

        return tokens

    # ── Private: Persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        """
        Save the BM25 index and corpus to disk.
        Called automatically after build().

        Two separate files:
          bm25.pkl        → BM25Okapi object (term statistics, IDF scores)
          bm25_corpus.pkl → List of MedDocument objects (full chunk data)

        Why two files?
          The BM25 object is small (~KB for typical medical docs).
          The corpus can be large (~MB for many PDFs).
          Separating them lets us inspect or update them independently.
        """
        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        # "wb" = write binary — pickle produces bytes, not text

        with open(self.corpus_path, "wb") as f:
            pickle.dump(self.corpus, f)

        logger.info(
            f"BM25 index saved → {self.index_path} "
            f"({self.index_path.stat().st_size // 1024} KB)"
        )

    def _load(self) -> None:
        """
        Load the BM25 index and corpus from disk.
        Called automatically in __init__ if index files exist.

        This means you don't need to re-run ingestion every time
        you restart the app — the index loads instantly from disk.
        """
        try:
            with open(self.index_path, "rb") as f:
                self.bm25 = pickle.load(f)
            # "rb" = read binary

            with open(self.corpus_path, "rb") as f:
                self.corpus = pickle.load(f)

            logger.info(
                f"BM25 index loaded from disk ✅ "
                f"({len(self.corpus)} docs, "
                f"vocab size: {len(self.bm25.idf)})"
            )
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}. Will rebuild on next ingestion.")
            self.bm25 = None
            self.corpus = None