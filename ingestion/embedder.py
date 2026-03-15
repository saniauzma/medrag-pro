# ingestion/embedder.py
# ---------------------
# Converts MedDocument chunks into dense vector embeddings.
#
# Model: BAAI/bge-m3 (local, via sentence-transformers)
#   - 1024-dimensional output vectors
#   - 8192 token max context
#   - State-of-the-art for retrieval as of 2024
#   - Fully local — no API calls, no cost
#
# Key design: asymmetric retrieval
#   Documents → embed raw text (with metadata prefix)
#   Queries   → embed with BGE query prefix
#   Using the same embedding path for both degrades retrieval quality.

from __future__ import annotations

import logging

import torch
# PyTorch — the backend that sentence-transformers runs on.
# We use it to check if CUDA (GPU) is available,
# and to switch the model to fp16 (half precision) on GPU.

from sentence_transformers import SentenceTransformer
# The library that wraps bge-m3.
# SentenceTransformer handles batching, tokenization,
# pooling, and normalization for us.

from config import settings
from ingestion.pdf_parser import MedDocument

logger = logging.getLogger(__name__)


class MedEmbedder:
    """
    Embeds MedDocument chunks using BAAI/bge-m3 locally.

    Two separate paths:
      embed_documents() → for indexing chunks into Qdrant
      embed_query()     → for embedding user questions at query time

    These MUST use different logic (asymmetric retrieval).
    Using embed_documents() for queries is a common bug.

    Usage:
        embedder = MedEmbedder()
        vectors = embedder.embed_documents(chunks)   # during ingestion
        vector  = embedder.embed_query("...")         # during retrieval
    """

    # BGE-M3 query prefix — required for asymmetric retrieval.
    # Defined as a class constant so it's easy to find and change.
    # This exact string comes from the BAAI/bge-m3 model card.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self):
        # ── Resolve device ────────────────────────────────────────
        # Determine actual device — never blindly trust settings.embedding_device.
        # If the user set "cuda" but CUDA isn't available, fail gracefully
        # instead of crashing with a confusing AssertionError.
        if settings.embedding_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif settings.embedding_device == "cuda":
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but not available — falling back to CPU. "
                    "If you have an NVIDIA GPU, reinstall torch with CUDA support: "
                    "uv add torch --index-url https://download.pytorch.org/whl/cu121"
                )
                device = "cpu"
            else:
                device = "cuda"
        else:
            device = "cpu"

        self.device = device
        logger.info(f"Embedding device: {self.device}")
        logger.info(f"Loading embedding model: {settings.embedding_model}")

        self.model = SentenceTransformer(
            settings.embedding_model,
            device=self.device,
        )

        # ── fp16 on CUDA only ─────────────────────────────────────
        # if self.device == "cuda":
        #     self.model = self.model.half()
        #     logger.info("fp16 enabled on CUDA — 2x speed, same quality")
        if self.device == "cuda":
            # Only use fp16 if we have enough VRAM (8GB+)
            # On 4GB cards, fp32 is safer — fp16 conversion itself needs extra memory
            total_vram = torch.cuda.get_device_properties(0).total_memory
            total_vram_gb = total_vram / (1024 ** 3)

            if total_vram_gb >= 7.0:
                self.model = self.model.half()
                logger.info(f"fp16 enabled ({total_vram_gb:.1f}GB VRAM)")
            else:
                logger.info(f"fp32 kept ({total_vram_gb:.1f}GB VRAM — fp16 skipped)")
        else:
            logger.info("Running on CPU — embedding will be slower")

        logger.info(f"Embedding model ready | dim={settings.embedding_dim}")


    # ── Public: Document Embedding ────────────────────────────────────────────

    def embed_documents(self, chunks: list[MedDocument]) -> list[list[float]]:
        """
        Embed a list of MedDocument chunks for indexing.

        Returns embeddings in the same order as input — index i of the
        output corresponds to index i of the input. This ordering is
        critical when we store (chunk, embedding) pairs in Qdrant.

        Each chunk is enriched with metadata before embedding
        (see _prepare_doc_text below).
        """
        if not chunks:
            return []

        # Prepare enriched text for each chunk
        texts = [self._prepare_doc_text(chunk) for chunk in chunks]
        # List comprehension → one enriched string per chunk

        logger.info(
            f"Embedding {len(texts)} chunks "
            f"(batch_size={settings.embedding_batch_size})"
        )

        embeddings = self.model.encode(
            texts,

            batch_size=settings.embedding_batch_size,
            # How many chunks to embed in one GPU forward pass.
            # 32 is safe for 8GB VRAM with bge-m3.
            # Larger batch = faster but more VRAM.
            # If you get CUDA out-of-memory errors, reduce this.

            show_progress_bar=True,
            # Prints a tqdm progress bar — useful for large ingestion jobs.
            # You'll see: "Batches: 100%|████| 8/8 [00:12<00:00,  1.54s/it]"

            normalize_embeddings=True,
            # L2-normalizes all vectors to unit length (magnitude = 1).
            # Critical for cosine similarity:
            #   normalized: cosine_sim(a, b) = dot_product(a, b)
            #   This lets Qdrant use fast dot product instead of
            #   computing full cosine similarity (faster at query time).

            convert_to_numpy=True,
            # Returns numpy arrays instead of PyTorch tensors.
            # Qdrant client expects Python lists or numpy arrays.
            # We convert to list below for JSON serialization compatibility.
        )

        # embeddings shape: (num_chunks, 1024)
        # .tolist() converts numpy array → Python list of lists
        return embeddings.tolist()

    # ── Public: Query Embedding ───────────────────────────────────────────────

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query for retrieval.

        IMPORTANT: This uses a different code path than embed_documents.
        The BGE query prefix is prepended here — never in embed_documents.

        Why single query at a time?
          At query time we embed one question from a user.
          Batching doesn't help for single inputs.
          Speed matters more here — users are waiting for a response.
        """
        # Prepend the BGE query prefix
        prefixed_query = self.QUERY_PREFIX + query
        # Example:
        # Input:  "What is the recommended dosage of metformin?"
        # Output: "Represent this sentence for searching relevant passages:
        #          What is the recommended dosage of metformin?"

        embedding = self.model.encode(
            prefixed_query,
            # Single string, not a list — model handles this fine

            normalize_embeddings=True,
            # Must normalize — same as documents so dot product = cosine sim

            convert_to_numpy=True,
        )
        # embedding shape: (1024,) — a 1D array for a single input

        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed raw text strings without any MedDocument structure.

        Used by the evaluation module to embed synthetic QA test sets
        where we have plain strings, not MedDocument objects.

        Also useful for quick experiments in a notebook.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    # ── Private ───────────────────────────────────────────────────────────────

    def _prepare_doc_text(self, chunk: MedDocument) -> str:
        """
        Enrich a chunk with metadata before embedding.

        Why prepend metadata to the text before embedding?

        The embedding model sees the FULL string as one input.
        By including title and section, the model encodes them
        into the same vector as the content.

        At query time, when someone asks:
          "What did the Results section say about response rates?"
        The query vector will be closer to chunks tagged with
        "Section: Results" than identical text tagged "Section: Methods".

        Example output:
          "Title: Efficacy of Drug X in Type 2 Diabetes |
           Section: Results |
           Response rate was 78% in the treatment group..."
        """
        parts = []

        if chunk.doc_title:
            parts.append(f"Title: {chunk.doc_title}")
            # Only add if non-empty — no "Title: " with blank value

        if chunk.section_title:
            parts.append(f"Section: {chunk.section_title}")

        # Always add the actual content last
        parts.append(chunk.content)

        return " | ".join(parts)
        # " | " separator is clean and unlikely to appear in medical text.
        # The model treats it as a soft boundary between metadata and content.

# ── Singleton instance ────────────────────────────────────────────────────────
# Module-level singleton — bge-m3 is loaded exactly once per process.
# Both IngestionPipeline and HybridRetriever import this same instance.
# This prevents loading the model twice and saves ~3.4GB VRAM.

_embedder_instance: MedEmbedder | None = None

def get_embedder() -> MedEmbedder:
    """
    Returns the shared MedEmbedder singleton.
    Creates it on first call, returns cached instance on all subsequent calls.

    Usage:
        from ingestion.embedder import get_embedder
        embedder = get_embedder()   # always returns same object
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MedEmbedder()
    return _embedder_instance