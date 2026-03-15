# ingestion/chunker.py
# --------------------
# Splits MedDocument objects into retrieval-ready chunks.
#
# Rules:
#   - TEXT chunks  → split by RecursiveCharacterTextSplitter
#   - TABLE chunks → never split (kept whole)
#   - IMAGE chunks → never split (caption is already short)
#
# All metadata from the parent MedDocument is fully preserved
# and copied into every child chunk.

from __future__ import annotations

import logging
from copy import deepcopy
# deepcopy creates a completely independent copy of an object.
# We use it to clone a MedDocument before modifying chunk_index.
# Without deepcopy, all child chunks would share the same object
# in memory — modifying one would modify all of them.

# NEW
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LangChain's most robust text splitter.
# "Recursive" means it tries separators in order:
#   first split on "\n\n", if chunks still too big try "\n", then ". ", then " "
# This preserves document structure much better than a naive character split.

from config import settings
from ingestion.pdf_parser import ContentType, MedDocument

logger = logging.getLogger(__name__)


class MedChunker:
    """
    Chunks MedDocument objects into retrieval-ready pieces.

    Only TEXT documents are split — tables and images are kept whole.
    Metadata is fully propagated to every child chunk.

    Usage:
        chunker = MedChunker()
        chunks = chunker.chunk(documents)
    """

    # Medical document separators — ordered from strongest to weakest boundary.
    # RecursiveCharacterTextSplitter tries each in order.
    # It only moves to the next separator if chunks are still too large.
    SEPARATORS = [
        "\n\n\n",   # major section break (strongest boundary — try first)
        "\n\n",     # paragraph break
        ".\n",      # sentence ending with newline (common in PDFs)
        ". ",       # regular sentence boundary
        ";\n",      # semicolon + newline (clinical lists)
        "\n",       # any newline
        " ",        # word boundary (last resort — least desirable split point)
    ]
    # Why this order matters:
    # If we split on " " first, we'd get random word fragments.
    # By trying paragraph breaks first, we keep related sentences together.
    # We only fall back to word-level splitting if absolutely necessary.

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            # Maximum size of each chunk.
            # We use approximate token count (see _token_len below).
            # 512 tokens ≈ 400 words ≈ 2-3 medical paragraphs.

            chunk_overlap=settings.chunk_overlap,
            # Number of tokens to repeat between consecutive chunks.
            # 64 tokens ≈ 2-3 sentences of overlap.
            # Ensures context isn't lost at chunk boundaries.

            separators=self.SEPARATORS,
            # Our custom medical-aware separators (defined above).

            length_function=self._token_len,
            # The function used to measure chunk size.
            # By default LangChain uses len() which counts characters.
            # We override it with our token estimator so chunk_size=512
            # means 512 tokens, not 512 characters.

            is_separator_regex=False,
            # Our separators are plain strings, not regex patterns.
            # Setting this False avoids any regex interpretation of
            # characters like "." which has special meaning in regex.
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(self, documents: list[MedDocument]) -> list[MedDocument]:
        """
        Takes raw parsed documents and returns chunked documents.

        TEXT → split into multiple chunks
        TABLE → returned as-is
        IMAGE → returned as-is

        Every chunk inherits the full metadata of its parent document.
        """
        chunked: list[MedDocument] = []

        for doc in documents:

            if doc.content_type in (ContentType.TABLE, ContentType.IMAGE):
                # ── Tables and images: never split ────────────
                # Just pass them through unchanged.
                # They already have correct metadata from the parser.
                chunked.append(doc)

            elif doc.content_type == ContentType.TEXT:
                # ── Text: split into chunks ───────────────────
                text_chunks = self.splitter.split_text(doc.content)
                # Returns a list of strings, each within chunk_size tokens.
                # Overlap is handled automatically by LangChain.

                for i, chunk_text in enumerate(text_chunks):
                    chunk_text = chunk_text.strip()
                    # .strip() removes leading/trailing whitespace
                    # that sometimes appears at split boundaries.

                    if len(chunk_text) < settings.min_chunk_size:
                        # Discard fragments that are too small.
                        # These are usually split artifacts like
                        # "et al." or "p < 0.05" in isolation.
                        continue

                    # deepcopy the parent document to preserve ALL metadata.
                    # This is much safer than constructing a new MedDocument
                    # manually — we can't accidentally forget a field.
                    new_doc = deepcopy(doc)

                    # Override only the fields that change per chunk:
                    new_doc.content = chunk_text
                    # The actual text content of this specific chunk.

                    new_doc.chunk_index = doc.chunk_index * 1000 + i
                    # Why * 1000 + i?
                    # doc.chunk_index = 5 (5th block on the page)
                    # i = 0, 1, 2 (sub-chunks within that block)
                    # Result: 5000, 5001, 5002
                    # This preserves reading order when sorting by chunk_index.
                    # If we just used i, chunks from different blocks would collide.

                    new_doc.metadata["is_sub_chunk"] = i > 0
                    # True for all chunks except the first.
                    # Useful for UI: "this is a continuation of a larger passage"

                    new_doc.metadata["sub_chunk_index"] = i
                    # Which sub-chunk within this block (0, 1, 2...)
                    # Helps reconstruct the full passage if needed.

                    chunked.append(new_doc)

        logger.info(
            f"Chunking complete: {len(documents)} documents → {len(chunked)} chunks"
        )
        return chunked

    def get_stats(self, chunks: list[MedDocument]) -> dict:
        """
        Returns chunking quality statistics.

        Call this after chunk() to verify your chunks look reasonable:
          - avg_tokens should be close to chunk_size (512)
          - min_tokens > min_chunk_size confirms filtering worked
          - max_tokens slightly above chunk_size is normal (overlap adds a few tokens)

        Usage:
            chunks = chunker.chunk(documents)
            stats = chunker.get_stats(chunks)
            print(stats)
            # {'total': 243, 'text': 198, 'tables': 31, 'images': 14,
            #  'avg_tokens': 487, 'min_tokens': 102, 'max_tokens': 541}
        """
        text_chunks  = [c for c in chunks if c.content_type == ContentType.TEXT]
        table_chunks = [c for c in chunks if c.content_type == ContentType.TABLE]
        image_chunks = [c for c in chunks if c.content_type == ContentType.IMAGE]

        token_lengths = [self._token_len(c.content) for c in text_chunks]
        # Only measure text chunks — tables/images aren't constrained by chunk_size

        return {
            "total_chunks":  len(chunks),
            "text_chunks":   len(text_chunks),
            "table_chunks":  len(table_chunks),
            "image_chunks":  len(image_chunks),
            "avg_tokens": round(sum(token_lengths) / len(token_lengths), 1)
                          if token_lengths else 0,
            "min_tokens": min(token_lengths) if token_lengths else 0,
            "max_tokens": max(token_lengths) if token_lengths else 0,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _token_len(self, text: str) -> int:
        """
        Approximate token count without loading a tokenizer.

        Rule of thumb: 1 token ≈ 4 characters for English text.
        This is the same approximation OpenAI uses in their docs.

        Why not use tiktoken or the HuggingFace tokenizer?
          1. Loading a tokenizer adds ~1s startup time just for chunking.
          2. For splitting decisions, exact token counts don't matter.
             Being off by 5-10% has no meaningful effect on retrieval quality.
          3. This function is called thousands of times during chunking —
             it must be fast.

        If you want exact token counts:
          from transformers import AutoTokenizer
          tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
          return len(tokenizer.encode(text))
        """
        return len(text) // 4