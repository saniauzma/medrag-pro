"""
embedder.py
-----------
Generates dense vector embeddings using BAAI/bge-m3 (local, free).
bge-m3 supports 1024-dim embeddings and handles long medical text well.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from config import settings

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """
    Wraps BAAI/bge-m3 for generating embeddings locally.

    bge-m3 advantages for medical RAG:
    - 1024-dim dense embeddings (high expressivity)
    - Handles up to 8192 tokens (great for long medical chunks)
    - State-of-the-art retrieval on medical benchmarks
    - Runs on GPU automatically if CUDA is available
    """

    def __init__(self, model_name: str = settings.embedding_model):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed_documents(
        self,
        docs: list[Document],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Embed a list of LangChain Documents.
        Returns list of float vectors, one per document.
        """
        texts = [doc.page_content for doc in docs]
        return self.embed_texts(texts, batch_size=batch_size, show_progress=show_progress)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Embed a list of raw strings."""
        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")

        # bge-m3 uses query/passage prefixes for better asymmetric retrieval
        # For indexing (passage), we add no prefix — bge-m3 handles this internally
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # cosine similarity ready
            convert_to_numpy=True,
        )

        logger.info(f"Embedding complete. Shape: {embeddings.shape}")
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.
        Uses the query instruction prefix for bge-m3 asymmetric retrieval.
        """
        # bge-m3 query instruction for better retrieval performance
        instructed_query = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode(
            instructed_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()
