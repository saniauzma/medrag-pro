# ingestion/pdf_parser.py
# -----------------------
# Parses medical PDF documents into structured MedDocument objects.
#
# Three content types are handled:
#   TEXT  → extracted via PyMuPDF block parsing
#   TABLE → extracted via Camelot (falls back to pdfplumber)
#   IMAGE → extracted via PyMuPDF, captioned via LLaVA-phi3 (Ollama)
#
# Output: List[MedDocument] — one object per content chunk

from __future__ import annotations
# Allows using type hints like list[MedDocument] before the class is defined.
# Without this, Python 3.9 and below would crash on forward references.
# Good habit even on 3.11 — makes the intent explicit.

import base64
# We need this to encode extracted images as base64 strings.
# Ollama's vision API expects images as base64, not raw bytes.

import io
# Provides in-memory byte streams.
# We use io.BytesIO to open image bytes as a PIL Image
# without saving to disk first.

import logging
# Standard Python logging — never use print() in production code.
# print() can't be filtered, redirected, or silenced easily.
# logging lets you control verbosity per module.

import re
# Regular expressions — used for extracting DOI numbers from text.
# DOIs follow a standard pattern: 10.XXXX/something

from dataclasses import dataclass, field
# dataclass auto-generates __init__, __repr__, __eq__ for us.
# We use it for MedDocument so we don't write boilerplate constructors.
# field(default_factory=dict) creates a fresh dict per instance —
# never use a mutable default like metadata={} directly in a dataclass.

from enum import Enum
# Enum for content type — safer than using raw strings like "text", "table".
# If you typo "tabel", an Enum catches it at definition time.
# A string "tabel" would silently pass through and cause bugs later.

from pathlib import Path
# Path() is the modern, OS-independent way to handle file paths.
# On Windows: Path("data/pdfs") works — no need for os.path.join().

from typing import Optional
# For type hints like Optional[str] = None (a value that might not exist).

import fitz
# fitz is the import name for PyMuPDF.
# PyMuPDF is the fastest Python PDF library — written in C.
# It handles text, image extraction, and page rendering.

import requests
# HTTP client — we use this to call Ollama's REST API for image captioning.

from PIL import Image
# Pillow — for opening image bytes and checking image dimensions.
# We use it to filter out tiny images (logos, icons) before sending to VLM.

from config import settings
# Our centralized settings singleton.
# All configurable values (model names, thresholds) come from here.


# ─── Content Type ─────────────────────────────────────────────────────────────

class ContentType(str, Enum):
    # str + Enum means the values ARE strings — ContentType.TEXT == "text" is True.
    # This makes serialization to JSON/dict trivial — no need to call .value everywhere.
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


# ─── MedDocument ──────────────────────────────────────────────────────────────

@dataclass
class MedDocument:
    """
    A single parsed chunk from a medical PDF.

    This is the core data structure of the entire pipeline.
    Every downstream component — chunker, embedder, vector store,
    retriever — works with MedDocument objects.

    Rich metadata is NOT optional for medical RAG:
      - section_title tells the LLM WHERE in the paper this came from
      - doi enables citation tracking back to the source
      - content_type tells the retriever how to handle each chunk differently
    """

    # ── Required fields ────────────────────────────────────────
    content: str
    # The actual text content of this chunk.
    # For TEXT: extracted paragraph text
    # For TABLE: markdown-formatted table string
    # For IMAGE: VLM-generated caption describing the figure

    content_type: ContentType
    # One of: ContentType.TEXT, ContentType.TABLE, ContentType.IMAGE
    # Used downstream to apply different chunking/retrieval strategies

    source_file: str
    # Original PDF filename e.g. "lancet_study_2024.pdf"
    # Stored in Qdrant metadata for citation display in UI

    page_number: int
    # 1-indexed page number in the original PDF
    # Shown to users: "Answer found on page 3 of lancet_study_2024.pdf"

    chunk_index: int
    # Position of this chunk within its page
    # Used to reconstruct reading order when displaying results

    # ── Optional metadata ──────────────────────────────────────
    section_title: str = ""
    # The nearest section heading above this chunk.
    # e.g. "Methods", "Results", "Discussion"
    # Injected into the embedding text to improve retrieval accuracy

    doc_title: str = ""
    # Full title of the paper — extracted from PDF metadata
    # Falls back to filename if metadata is missing

    doc_authors: str = ""
    # Author list — extracted from PDF metadata
    # Used in citations: "Smith et al. (2024)"

    doi: str = ""
    # Digital Object Identifier — unique ID for the paper
    # e.g. "10.1016/S0140-6736(24)00123-4"
    # Enables linking back to the original source

    image_path: str = ""
    # Only populated for IMAGE chunks
    # Absolute path to the saved image file on disk
    # Used by the UI to display the actual figure alongside the caption

    metadata: dict = field(default_factory=dict)
    # Flexible catch-all for any extra metadata.
    # We use it for: table accuracy score, image dimensions, sub-chunk index.
    # field(default_factory=dict) is REQUIRED for mutable defaults in dataclasses.
    # If you wrote metadata: dict = {} Python would share one dict across all instances.

    def to_dict(self) -> dict:
        """
        Serialize to a flat dictionary.
        Used when storing metadata in Qdrant — Qdrant only accepts flat dicts,
        not nested objects.
        """
        return {
            "content": self.content,
            "content_type": self.content_type.value,  # .value gives "text" not ContentType.TEXT
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "section_title": self.section_title,
            "doc_title": self.doc_title,
            "doc_authors": self.doc_authors,
            "doi": self.doi,
            "image_path": self.image_path,
            **self.metadata,
            # ** unpacks the dict — extra metadata keys land at the top level.
            # e.g. {"table_accuracy": 0.99} becomes a top-level key in the output.
        }

# ─── Parser ───────────────────────────────────────────────────────────────────

class MedPDFParser:
    """
    Parses medical PDFs into structured MedDocument objects.

    Handles three content types:
      - Text (PyMuPDF block parsing with heading detection)
      - Tables (Camelot lattice → pdfplumber fallback)
      - Images (PyMuPDF extraction → LLaVA-phi3 captioning)

    Usage:
        parser = MedPDFParser()
        docs = parser.parse("data/pdfs/study.pdf")
    """

    def __init__(self):
        # Pull all settings from the singleton — no hardcoded values here
        self.extract_tables = settings.extract_tables
        self.extract_images = settings.extract_images
        self.min_image_size = settings.min_image_size
        self.table_flavor = settings.table_flavor

        # Set up module-level logger
        # __name__ gives "ingestion.pdf_parser" — useful for filtering logs
        self.logger = logging.getLogger(__name__)

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, pdf_path: str | Path) -> list[MedDocument]:
        """
        Parse a single PDF file.
        Returns a list of MedDocument objects — one per content chunk.

        str | Path means you can pass either:
          parser.parse("data/pdfs/study.pdf")    # string
          parser.parse(Path("data/pdfs/study.pdf"))  # Path object
        Both work.
        """
        pdf_path = Path(pdf_path)
        # Always convert to Path immediately — gives us .exists(), .name, .stem etc.

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        # Fail loudly and early with a clear message.
        # Better than a cryptic fitz error 10 lines later.

        self.logger.info(f"Parsing: {pdf_path.name}")
        docs: list[MedDocument] = []

        # Extract PDF-level metadata once — shared across all chunks from this PDF
        meta = self._extract_pdf_metadata(pdf_path)

        # ── Step 1: Text ──────────────────────────────────────
        docs.extend(self._extract_text(pdf_path, meta))
        self.logger.info(
            f"  Text chunks: {sum(1 for d in docs if d.content_type == ContentType.TEXT)}"
        )

        # ── Step 2: Tables ────────────────────────────────────
        if self.extract_tables:
            table_docs = self._extract_tables(pdf_path, meta)
            docs.extend(table_docs)
            self.logger.info(f"  Table chunks: {len(table_docs)}")

        # ── Step 3: Images ────────────────────────────────────
        if self.extract_images:
            image_docs = self._extract_images(pdf_path, meta)
            docs.extend(image_docs)
            self.logger.info(f"  Image chunks: {len(image_docs)}")

        self.logger.info(f"  Total: {len(docs)} chunks from {pdf_path.name}")
        return docs

    def parse_directory(self, pdf_dir: str | Path) -> list[MedDocument]:
        """
        Parse all PDFs in a folder.
        Skips files that fail — logs the error but continues.
        This is important for batch ingestion — one bad PDF shouldn't
        stop processing 50 others.
        """
        pdf_dir = Path(pdf_dir)
        all_docs = []
        pdfs = list(pdf_dir.glob("*.pdf"))
        # .glob("*.pdf") finds all PDFs in the directory (not recursive)

        self.logger.info(f"Found {len(pdfs)} PDFs in {pdf_dir}")

        for pdf in pdfs:
            try:
                all_docs.extend(self.parse(pdf))
            except Exception as e:
                # Log and continue — don't let one bad PDF crash everything
                self.logger.error(f"Failed to parse {pdf.name}: {e}")

        return all_docs

    # ── Private: PDF Metadata ─────────────────────────────────────────────────

    def _extract_pdf_metadata(self, pdf_path: Path) -> dict:
        """
        Extract title, authors, and DOI from the PDF.

        PDFs store metadata in two places:
          1. The PDF header (accessed via doc.metadata) — often empty or wrong
          2. The actual text on the first page — more reliable for papers

        We try both and take the best result.
        """
        meta = {
            "doc_title": pdf_path.stem,  # fallback: use filename without extension
            "doc_authors": "",
            "doi": "",
        }

        try:
            doc = fitz.open(str(pdf_path))
            # fitz.open requires a string path, not a Path object

            pdf_meta = doc.metadata
            # Returns a dict like: {"title": "...", "author": "...", "subject": "..."}

            if pdf_meta.get("title"):
                meta["doc_title"] = pdf_meta["title"].strip()

            if pdf_meta.get("author"):
                meta["doc_authors"] = pdf_meta["author"].strip()

            # DOI is rarely in PDF header metadata — search first 2 pages of text
            for page_num in range(min(2, len(doc))):
                text = doc[page_num].get_text()
                doi = self._extract_doi(text)
                if doi:
                    meta["doi"] = doi
                    break
                # break after finding first DOI — papers only have one

            doc.close()
            # Always close fitz documents — they hold file handles

        except Exception as e:
            self.logger.warning(f"Could not extract PDF metadata: {e}")
            # Non-fatal — we still have the filename fallback

        return meta

    def _extract_doi(self, text: str) -> str:
        """
        Extract DOI from text using regex.

        DOI format: 10.XXXX/anything
        Examples:
          10.1016/S0140-6736(24)00123-4
          10.1056/NEJMoa2304013
          10.1038/s41591-023-02619-z
        """
        # re.IGNORECASE because some PDFs write "DOI:" or "doi:"
        pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(0) if match else ""

# ── Private: Text Extraction ──────────────────────────────────────────────

    def _extract_text(self, pdf_path: Path, meta: dict) -> list[MedDocument]:
        """
        Extract text from each page using PyMuPDF's block-level parsing.

        Why block-level instead of page.get_text()?
          page.get_text() gives you one giant string per page — no structure.
          Block-level parsing gives you individual paragraphs with font metadata.
          Font metadata lets us detect headings — critical for section tracking.

        Section tracking matters because:
          A chunk from the "Results" section has very different meaning
          than the same sentence in "Limitations".
          We store this in section_title for every chunk.
        """
        SKIP_SECTIONS = {
            "references", "bibliography", "acknowledgements",
            "acknowledgments", "funding", "conflicts of interest",
            "author contributions", "supplementary",
        }
        docs = []
        doc = fitz.open(str(pdf_path))

        current_section = ""
        # Tracks the most recently seen heading.
        # Updates as we scan down through blocks.
        # Every text chunk gets tagged with the current section.

        chunk_index = 0

        for page_num in range(len(doc)):
            page = doc[page_num]

            # get_text("dict") returns structured data:
            # {"blocks": [{"type": 0, "lines": [{"spans": [...]}]}]}
            # type 0 = text block, type 1 = image block
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block["type"] != 0:
                    # Skip image blocks — we handle those separately in _extract_images
                    continue

                block_text = ""
                is_heading = False

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        # A span is the smallest text unit — same font, size, color
                        text = span["text"].strip()
                        if not text:
                            continue

                        font_size = span.get("size", 12)
                        # Default 12 if missing — body text is usually 10-12pt

                        font_flags = span.get("flags", 0)
                        # flags is a bitmask. Bit 4 (2**4 = 16) = bold
                        is_bold = bool(font_flags & 2**4)

                        # Heading detection heuristic:
                        # Font size >= 13 → definitely a heading
                        # Bold + size >= 11 → likely a subheading
                        # These thresholds work for most medical papers.
                        # Not perfect — but good enough for metadata tagging.
                        if font_size >= 13 or (is_bold and font_size >= 11):
                            current_section = text
                            is_heading = True
                        else:
                            block_text += text + " "

                block_text = block_text.strip()

                # Discard chunks that are too short — likely headers, page numbers,
                # figure labels ("Fig. 1"), or other noise
                if len(block_text) < settings.min_chunk_size:
                    continue

                if current_section.lower() in SKIP_SECTIONS:
                    continue

                docs.append(MedDocument(
                    content=block_text,
                    content_type=ContentType.TEXT,
                    source_file=pdf_path.name,
                    page_number=page_num + 1,   # fitz is 0-indexed, humans expect 1-indexed
                    chunk_index=chunk_index,
                    section_title=current_section,
                    doc_title=meta["doc_title"],
                    doc_authors=meta["doc_authors"],
                    doi=meta["doi"],
                ))
                chunk_index += 1

        doc.close()
        return docs

# ── Private: Table Extraction ─────────────────────────────────────────────

    def _extract_tables(self, pdf_path: Path, meta: dict) -> list[MedDocument]:
        """
        Extract tables from PDF.

        Strategy:
          1. Try Camelot (lattice mode) — best for bordered tables in medical papers
          2. If Camelot fails, fall back to pdfplumber

        Why two libraries?
          Camelot is more accurate but requires Ghostscript to be installed.
          pdfplumber is pure Python — always works but less accurate for complex tables.
          Having a fallback makes the pipeline robust.

        Why convert tables to markdown?
          LLMs understand markdown tables natively.
          "| Drug | Dosage | Effect |" is more meaningful to an LLM
          than a raw CSV or a flattened string.
        """
        docs = []

        # ── Attempt 1: Camelot ────────────────────────────────
        try:
            import camelot

            tables = camelot.read_pdf(
                str(pdf_path),
                pages="all",        # extract from all pages
                flavor=self.table_flavor,
                # "lattice" uses table borders to detect cells
                # More accurate for formal medical paper tables
            )

            for i, table in enumerate(tables):
                df = table.df
                # table.df is a pandas DataFrame of the extracted table

                if df.empty:
                    continue

                markdown_table = self._df_to_markdown(df)

                docs.append(MedDocument(
                    content=f"[TABLE]\n{markdown_table}",
                    # [TABLE] prefix helps the LLM understand this is structured data
                    content_type=ContentType.TABLE,
                    source_file=pdf_path.name,
                    page_number=table.page,
                    chunk_index=i,
                    doc_title=meta["doc_title"],
                    doc_authors=meta["doc_authors"],
                    doi=meta["doi"],
                    metadata={
                        "table_accuracy": table.accuracy
                        # Camelot scores its own extraction confidence (0-100)
                        # We store this — useful for filtering low-quality extractions
                    },
                ))

            if docs:
                # Camelot succeeded — skip pdfplumber
                return docs

        except Exception as e:
            self.logger.warning(f"Camelot failed: {e}. Trying pdfplumber...")

        # ── Attempt 2: pdfplumber fallback ────────────────────
        try:
            import pdfplumber
            import pandas as pd

            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    for i, table in enumerate(page.extract_tables()):
                        if not table or len(table) < 2:
                            # Need at least a header row + one data row
                            continue

                        # table is a list of lists — first row is headers
                        df = pd.DataFrame(table[1:], columns=table[0])
                        markdown_table = self._df_to_markdown(df)

                        docs.append(MedDocument(
                            content=f"[TABLE]\n{markdown_table}",
                            content_type=ContentType.TABLE,
                            source_file=pdf_path.name,
                            page_number=page_num + 1,
                            chunk_index=i,
                            doc_title=meta["doc_title"],
                            doc_authors=meta["doc_authors"],
                            doi=meta["doi"],
                        ))

        except Exception as e:
            self.logger.error(f"pdfplumber also failed: {e}")

        return docs

    def _df_to_markdown(self, df) -> str:
        """
        Convert a pandas DataFrame to a markdown table string.

        Example output:
          | Drug     | Dose  | Response Rate |
          |----------|-------|---------------|
          | Drug A   | 50mg  | 78%           |
          | Drug B   | 100mg | 65%           |
        """
        try:
            return df.to_markdown(index=False)
            # tabulate library provides this — already in requirements
        except Exception:
            # Manual fallback if tabulate isn't installed
            rows = [" | ".join(str(c) for c in df.columns)]
            rows.append(" | ".join(["---"] * len(df.columns)))
            for _, row in df.iterrows():
                rows.append(" | ".join(str(v) for v in row))
            return "\n".join(rows)


# ── Private: Image Extraction ─────────────────────────────────────────────

    def _extract_images(self, pdf_path: Path, meta: dict) -> list[MedDocument]:
        """
        Extract images from the PDF and caption them with LLaVA-phi3.

        Why caption instead of embedding the image directly?
          Most vector stores (including Qdrant) store vectors of fixed dimension.
          Text embeddings and image embeddings live in different spaces.
          Converting images to text captions lets us store everything in one
          unified vector index — much simpler architecture.

        The trade-off:
          We lose some visual detail in translation.
          But LLaVA-phi3 is surprisingly good at describing medical figures,
          MRI scans, survival curves, and flowcharts.
        """
        docs = []

        # Create a folder to save extracted images
        # Structure: data/cache/images/<pdf_name>/page1_img0.png
        image_dir = settings.cache_dir / "images" / pdf_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))

        for page_num in range(len(doc)):
            page = doc[page_num]

            # get_images(full=True) returns a list of tuples:
            # (xref, smask, width, height, bpc, colorspace, alt, name, filter, referencer)
            # We only need xref — the image reference ID
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    # extract_image returns a dict:
                    # {"image": bytes, "ext": "png", "width": 800, "height": 600, ...}
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_ext = base_image["ext"]

                    # Check image dimensions — filter out tiny decorative elements
                    img = Image.open(io.BytesIO(image_bytes))
                    # io.BytesIO wraps bytes as a file-like object
                    # PIL can then open it without writing to disk
                    w, h = img.size

                    if w < self.min_image_size or h < self.min_image_size:
                        # Skip logos, horizontal rules, small icons
                        continue

                    # Save image to disk for UI display later
                    img_filename = f"page{page_num + 1}_img{img_index}.{img_ext}"
                    img_path = image_dir / img_filename
                    img_path.write_bytes(image_bytes)

                    # Caption the image using LLaVA via Ollama
                    try:
                        caption = self._caption_image(image_bytes)
                    except Exception as caption_err:
                        self.logger.warning(f"Captioning skipped: {caption_err}")
                        caption = "Medical figure — captioning unavailable"

                    docs.append(MedDocument(
                        content=f"[FIGURE] {caption}",
                        # [FIGURE] prefix helps LLM know this came from a visual
                        content_type=ContentType.IMAGE,
                        source_file=pdf_path.name,
                        page_number=page_num + 1,
                        chunk_index=img_index,
                        doc_title=meta["doc_title"],
                        doc_authors=meta["doc_authors"],
                        doi=meta["doi"],
                        image_path=str(img_path),
                        metadata={"image_size": f"{w}x{h}"},
                    ))

                except Exception as e:
                    self.logger.warning(
                        f"Skipping image {img_index} on page {page_num + 1}: {e}"
                    )
                    # Non-fatal — skip this image, continue with others

        doc.close()
        return docs

    def _caption_image(self, image_bytes: bytes) -> str:
        """
        Send image to LLaVA-phi3 via Ollama's REST API and get a caption.

        Why REST API instead of the ollama Python client?
          The ollama Python client doesn't support images yet.
          The REST API does — we call it directly with requests.

        The prompt is carefully written for medical images:
          - Asks for figure TYPE (graph, MRI, flowchart, etc.)
          - Asks for KEY FINDINGS visible in the image
          - Asks for LABELS and AXES if present
          - Requests medical terminology
        """
        # Ollama expects images as base64-encoded strings
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        # base64.b64encode → bytes → .decode("utf-8") → string

        prompt = (
            "You are analyzing a figure from a medical research paper. "
            "Describe this image in detail: "
            "1) What type of figure is it? (e.g. bar chart, Kaplan-Meier curve, "
            "MRI scan, microscopy image, flowchart, forest plot, etc.) "
            "2) What does it show? Include axis labels, legend items, and key values if visible. "
            "3) What is the main finding or conclusion that can be drawn from this figure? "
            "Use precise medical and scientific terminology."
        )

        try:
            response = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.vlm_model,
                    "prompt": prompt,
                    "images": [b64_image],  # list of base64 strings
                    "stream": False,         # wait for full response, don't stream
                    "options": {
                        "temperature": 0.1,
                        # Low temperature for factual image description.
                        # Not 0.0 because vision tasks benefit from slight flexibility.
                    },
                },
                timeout=10,
                # 60 second timeout — VLM inference can be slow on CPU
                # Increase this if your machine is slow
            )
            response.raise_for_status()
            # raise_for_status() throws an exception if status code is 4xx or 5xx

            return response.json().get("response", "").strip()

        except Exception as e:
            self.logger.warning(f"VLM captioning failed: {e}")
            return "Medical figure — captioning unavailable"
            # Return a placeholder so the document is still indexed
            # A chunk with a placeholder is better than no chunk at all


# ─── CLI Quick Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run this directly to test the parser on a single PDF:
    # python -m ingestion.pdf_parser data/pdfs/your_paper.pdf
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.pdf_parser <path/to/paper.pdf>")
        sys.exit(1)

    parser = MedPDFParser()
    documents = parser.parse(sys.argv[1])

    print(f"\n{'='*60}")
    print(f"Parsed {len(documents)} chunks")
    print(f"{'='*60}")

    for doc in documents[:5]:  # preview first 5 chunks
        print(f"\n[{doc.content_type.value.upper()}] "
              f"Page {doc.page_number} | Section: '{doc.section_title}'")
        print(f"  {doc.content[:200]}...")