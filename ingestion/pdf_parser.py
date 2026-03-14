"""
pdf_parser.py
-------------
Extracts clean text, tables, and images from medical PDFs.
Produces a list of RawPage objects with full metadata.
"""

import fitz  # PyMuPDF
import base64
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import pdfplumber

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class RawImage:
    page_number: int
    image_index: int
    base64_data: str        # base64-encoded PNG
    width: int
    height: int
    caption: str = ""       # filled later by VLM


@dataclass
class RawTable:
    page_number: int
    table_index: int
    markdown: str           # table converted to markdown string
    raw_data: list          # raw list-of-lists from pdfplumber


@dataclass
class RawPage:
    page_number: int
    text: str
    tables: list[RawTable] = field(default_factory=list)
    images: list[RawImage] = field(default_factory=list)
    section_header: Optional[str] = None
    source_file: str = ""


# ── PDF Parser ────────────────────────────────────────────────────────────────

class PDFParser:
    """
    Parses a PDF into RawPage objects containing:
    - clean text (with section headers detected)
    - tables as markdown (via pdfplumber)
    - images as base64 PNGs (for VLM captioning downstream)
    """

    # Minimum image dimensions to bother extracting (filters out icons/logos)
    MIN_IMAGE_WIDTH = 100
    MIN_IMAGE_HEIGHT = 100

    def __init__(self, extract_images: bool = True, extract_tables: bool = True):
        self.extract_images = extract_images
        self.extract_tables = extract_tables

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, pdf_path: str | Path) -> list[RawPage]:
        """Parse a PDF file and return a list of RawPage objects."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing: {pdf_path.name}")

        raw_pages: list[RawPage] = []

        # Open with both PyMuPDF (text + images) and pdfplumber (tables)
        with fitz.open(str(pdf_path)) as mupdf_doc, \
             pdfplumber.open(str(pdf_path)) as plumber_doc:

            for page_num in range(len(mupdf_doc)):
                mupdf_page = mupdf_doc[page_num]
                plumber_page = plumber_doc.pages[page_num]

                text = self._extract_text(mupdf_page)
                tables = self._extract_tables(plumber_page, page_num) if self.extract_tables else []
                images = self._extract_images(mupdf_page, mupdf_doc, page_num) if self.extract_images else []
                header = self._detect_section_header(mupdf_page)

                raw_pages.append(RawPage(
                    page_number=page_num + 1,   # 1-indexed for human readability
                    text=text,
                    tables=tables,
                    images=images,
                    section_header=header,
                    source_file=pdf_path.name,
                ))

        logger.info(
            f"Parsed {len(raw_pages)} pages | "
            f"{sum(len(p.tables) for p in raw_pages)} tables | "
            f"{sum(len(p.images) for p in raw_pages)} images"
        )
        return raw_pages

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _extract_text(self, page: fitz.Page) -> str:
        """Extract and clean text from a PyMuPDF page."""
        # "text" flag extracts plain text preserving layout
        text = page.get_text("text")

        # Basic cleaning
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]   # drop blank lines

        return "\n".join(lines)

    def _extract_tables(self, page: pdfplumber.page.Page, page_num: int) -> list[RawTable]:
        """Extract tables from a pdfplumber page and convert to markdown."""
        tables = []
        try:
            raw_tables = page.extract_tables()
            for i, raw in enumerate(raw_tables):
                if not raw:
                    continue
                markdown = self._table_to_markdown(raw)
                tables.append(RawTable(
                    page_number=page_num + 1,
                    table_index=i,
                    markdown=markdown,
                    raw_data=raw,
                ))
        except Exception as e:
            logger.warning(f"Table extraction failed on page {page_num + 1}: {e}")
        return tables

    def _extract_images(
        self,
        page: fitz.Page,
        doc: fitz.Document,
        page_num: int,
    ) -> list[RawImage]:
        """Extract images as base64 PNGs from a PyMuPDF page."""
        images = []
        try:
            image_list = page.get_images(full=True)
            for i, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                w, h = base_image["width"], base_image["height"]

                # Skip tiny images (icons, bullets, etc.)
                if w < self.MIN_IMAGE_WIDTH or h < self.MIN_IMAGE_HEIGHT:
                    continue

                image_bytes = base_image["image"]
                b64 = base64.b64encode(image_bytes).decode("utf-8")

                images.append(RawImage(
                    page_number=page_num + 1,
                    image_index=i,
                    base64_data=b64,
                    width=w,
                    height=h,
                ))
        except Exception as e:
            logger.warning(f"Image extraction failed on page {page_num + 1}: {e}")
        return images

    def _detect_section_header(self, page: fitz.Page) -> Optional[str]:
        """
        Detect the dominant section header on a page using font-size heuristics.
        Returns the largest bold/heading text found, or None.
        """
        blocks = page.get_text("dict")["blocks"]
        candidates = []

        for block in blocks:
            if block.get("type") != 0:   # type 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2**4)   # bit 4 = bold
                    size = span.get("size", 0)
                    text = span.get("text", "").strip()

                    if text and is_bold and size > 11:
                        candidates.append((size, text))

        if not candidates:
            return None

        # Return the text with the largest font size
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """Convert a pdfplumber raw table (list of lists) to a markdown string."""
        if not table:
            return ""

        # Clean None values
        def clean(val):
            if val is None:
                return ""
            return str(val).strip().replace("\n", " ")

        rows = [[clean(cell) for cell in row] for row in table]

        # First row is the header
        header = rows[0]
        separator = ["---"] * len(header)
        body = rows[1:]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in body:
            # Pad row if it has fewer columns than header
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
