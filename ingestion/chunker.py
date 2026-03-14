"""
chunker.py
----------
Converts RawPage objects into LangChain Documents with:
- Semantic chunking (sentence-aware boundaries)
- Rich metadata (page, section, source, content_type)
- Separate handling for text, tables, and image captions
"""

import re
import logging
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.pdf_parser import RawPage
from config import settings

logger = logging.getLogger(__name__)


class MedicalChunker:
    """
    Produces LangChain Documents from RawPage objects.

    Strategy:
    - TEXT: RecursiveCharacterTextSplitter with sentence-aware separators
    - TABLES: Each table = one Document (never split tables mid-row)
    - IMAGES: Each image caption = one Document
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",        # paragraph boundary
                "\n",          # line boundary
                ". ",          # sentence boundary
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                "",
            ],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, pages: list[RawPage]) -> list[Document]:
        """
        Convert all RawPage objects into a flat list of LangChain Documents.
        """
        docs: list[Document] = []

        for page in pages:
            docs.extend(self._chunk_text(page))
            docs.extend(self._chunk_tables(page))
            docs.extend(self._chunk_images(page))

        logger.info(
            f"Chunked {len(pages)} pages → {len(docs)} documents "
            f"(text + tables + image captions)"
        )
        return docs

    # ── Text Chunking ─────────────────────────────────────────────────────────

    def _chunk_text(self, page: RawPage) -> list[Document]:
        """Split page text into overlapping chunks with metadata."""
        text = page.text.strip()
        if not text:
            return []

        # Remove table-like content from text to avoid duplication
        # (pdfplumber already extracted tables separately)
        text = self._remove_table_artifacts(text)
        if not text.strip():
            return []

        chunks = self.splitter.split_text(text)
        docs = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": page.source_file,
                    "page_number": page.page_number,
                    "section_header": page.section_header or "Unknown",
                    "content_type": "text",
                    "chunk_index": i,
                    # chunk_id used for deduplication / citation
                    "chunk_id": f"{page.source_file}::p{page.page_number}::t{i}",
                },
            ))
        return docs

    # ── Table Chunking ────────────────────────────────────────────────────────

    def _chunk_tables(self, page: RawPage) -> list[Document]:
        """Each table becomes its own Document — never split."""
        docs = []
        for table in page.tables:
            if not table.markdown.strip():
                continue

            # Prepend context: section header + table label
            header_context = ""
            if page.section_header:
                header_context = f"Section: {page.section_header}\n\n"

            content = (
                f"{header_context}"
                f"[TABLE from page {page.page_number}]\n\n"
                f"{table.markdown}"
            )

            docs.append(Document(
                page_content=content,
                metadata={
                    "source": page.source_file,
                    "page_number": page.page_number,
                    "section_header": page.section_header or "Unknown",
                    "content_type": "table",
                    "table_index": table.table_index,
                    "chunk_id": f"{page.source_file}::p{page.page_number}::tbl{table.table_index}",
                },
            ))
        return docs

    # ── Image Caption Chunking ────────────────────────────────────────────────

    def _chunk_images(self, page: RawPage) -> list[Document]:
        """Each image caption becomes its own Document."""
        docs = []
        for image in page.images:
            caption = image.caption.strip()
            if not caption or caption.startswith("[Image"):
                continue   # skip failed captions

            content = (
                f"[FIGURE from page {image.page_number}]\n\n"
                f"{caption}"
            )

            docs.append(Document(
                page_content=content,
                metadata={
                    "source": page.source_file,
                    "page_number": image.page_number,
                    "section_header": page.section_header or "Unknown",
                    "content_type": "figure",
                    "image_index": image.image_index,
                    "image_dimensions": f"{image.width}x{image.height}",
                    "chunk_id": f"{page.source_file}::p{image.page_number}::img{image.image_index}",
                },
            ))
        return docs

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _remove_table_artifacts(text: str) -> str:
        """
        Remove patterns that look like table rows left in the text stream.
        These are already captured by pdfplumber.
        """
        # Remove lines that are purely numbers/pipes/dashes (table-like)
        lines = text.splitlines()
        filtered = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are mostly table separators or pure numeric rows
            if re.match(r'^[\|\-\+\s\d\.,%]+$', stripped) and len(stripped) > 5:
                continue
            filtered.append(line)
        return "\n".join(filtered)
