"""
loader.py — Corpus document loader for AI Act Navigator

Handles two source types:
  1. AI Act HTML  — fetched live from EUR-Lex (stable public URL, reproducible)
  2. PDF files    — read from data/raw/ (GPAI CoP chapters + GPAI Guidelines)

Each loaded document is returned as a LoadedDocument dataclass containing:
  - raw text content
  - source metadata (id, title, doc_type, url/path, language)

This metadata flows through the entire pipeline and ends up on every Qdrant chunk.
"""

import re
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
import fitz  # pymupdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EUR_LEX_HTML_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
    "?uri=CELEX:32024R1689"
)

# Maps each local PDF filename to its document identity metadata
PDF_REGISTRY: dict[str, dict] = {
    "ai_act_2024_1689.pdf": {
        "doc_id": "ai_act",
        "title": "Regulation (EU) 2024/1689 — AI Act",
        "doc_type": "regulation",
        "source_org": "EUR-Lex",
        "date_published": "2024-07-12",
        "is_normative": True,
    },
    "gpai_cop_chapter1_transparency.pdf": {
        "doc_id": "gpai_cop_transparency",
        "title": "GPAI Code of Practice — Chapter 1: Transparency",
        "doc_type": "code_of_practice",
        "source_org": "EU AI Office",
        "date_published": "2025-07-10",
        "is_normative": False,
    },
    "gpai_cop_chapter2_copyright.pdf": {
        "doc_id": "gpai_cop_copyright",
        "title": "GPAI Code of Practice — Chapter 2: Copyright",
        "doc_type": "code_of_practice",
        "source_org": "EU AI Office",
        "date_published": "2025-07-10",
        "is_normative": False,
    },
    "gpai_cop_chapter3_safety_security.pdf": {
        "doc_id": "gpai_cop_safety",
        "title": "GPAI Code of Practice — Chapter 3: Safety and Security",
        "doc_type": "code_of_practice",
        "source_org": "EU AI Office",
        "date_published": "2025-07-10",
        "is_normative": False,
    },
    "gpai_guidelines_commission_2025.pdf": {
        "doc_id": "gpai_guidelines",
        "title": "Guidelines on GPAI Model Obligations — European Commission",
        "doc_type": "guidelines",
        "source_org": "European Commission",
        "date_published": "2025-07-18",
        "is_normative": False,
    },
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LoadedDocument:
    """A single loaded source document, ready for chunking."""
    doc_id: str
    title: str
    doc_type: str               # "regulation" | "code_of_practice" | "guidelines"
    source_org: str
    date_published: str
    is_normative: bool
    text: str                   # full extracted text
    source_url: Optional[str] = None
    source_path: Optional[str] = None
    load_method: str = "unknown"  # "html_fetch" | "pdf_extract"
    char_count: int = field(init=False)
    word_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())

    def __repr__(self) -> str:
        return (
            f"LoadedDocument(id={self.doc_id!r}, "
            f"words={self.word_count:,}, "
            f"method={self.load_method!r})"
        )


# ---------------------------------------------------------------------------
# HTML loader — AI Act from EUR-Lex
# ---------------------------------------------------------------------------

def load_ai_act_html(
    url: str = EUR_LEX_HTML_URL,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> LoadedDocument:
    """
    Fetch the AI Act HTML from EUR-Lex and extract clean text.

    Uses BeautifulSoup to strip navigation, headers, and footers,
    preserving only the normative content with its structural markers
    (article headings, paragraph numbers, annex titles).

    Args:
        url: EUR-Lex HTML URL for the AI Act
        retries: number of retry attempts on network failure
        retry_delay: seconds between retries

    Returns:
        LoadedDocument with extracted text and source metadata
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; AI-Act-Navigator/1.0; "
            "research/non-commercial)"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    html_content = None
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                f"Fetching AI Act HTML from EUR-Lex "
                f"(attempt {attempt}/{retries})"
            )
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            html_content = response.text
            logger.info(f"Fetched {len(html_content):,} characters of HTML")
            break
        except requests.RequestException as e:
            last_error = e
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(retry_delay)

    if html_content is None:
        raise RuntimeError(
            f"Failed to fetch AI Act HTML after {retries} attempts. "
            f"Last error: {last_error}"
        )

    text = _extract_text_from_html(html_content)

    return LoadedDocument(
        doc_id="ai_act",
        title="Regulation (EU) 2024/1689 — AI Act",
        doc_type="regulation",
        source_org="EUR-Lex",
        date_published="2024-07-12",
        is_normative=True,
        text=text,
        source_url=url,
        load_method="html_fetch",
    )


def _extract_text_from_html(html: str) -> str:
    """
    Extract clean normative text from EUR-Lex HTML.

    EUR-Lex HTML structure:
    - Main content lives in <div id="document-content"> or similar
    - Navigation, breadcrumbs, and footers are in separate divs
    - Article headings use <p class="oj-ti-art"> pattern
    - Paragraph numbers use <p class="oj-normal"> pattern

    We preserve structural markers by converting key HTML elements
    to plain-text equivalents before stripping tags.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Remove EUR-Lex navigation divs
    for div in soup.find_all("div", class_=lambda c: c and any(
        kw in c for kw in [
            "navigation", "breadcrumb", "language",
            "metadata", "toolbar", "sidebar",
        ]
    )):
        div.decompose()

    # Preserve structural markers as plain-text anchors
    for tag in soup.find_all(
        class_=lambda c: c and "ti-art" in str(c)
    ):
        tag.insert_before("\n\n### ")
        tag.insert_after("\n")

    for tag in soup.find_all(
        class_=lambda c: c and "ti-annex" in str(c)
    ):
        tag.insert_before("\n\n## ANNEX: ")
        tag.insert_after("\n")

    # Extract plain text
    text = soup.get_text(separator="\n")

    # Clean up whitespace while preserving paragraph structure
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()
    logger.info(
        f"Extracted {len(text):,} characters of clean text from HTML"
    )
    return text


# ---------------------------------------------------------------------------
# PDF loader
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: Path) -> LoadedDocument:
    """
    Extract text from a registered PDF using PyMuPDF (fitz).

    PyMuPDF preserves layout better than pdfplumber for legal documents —
    it handles column structure and doesn't merge footnotes with body text.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        LoadedDocument with extracted text and source metadata

    Raises:
        FileNotFoundError: if the PDF does not exist
        KeyError: if the filename is not in PDF_REGISTRY
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = pdf_path.name
    if filename not in PDF_REGISTRY:
        raise KeyError(
            f"Unknown PDF: {filename!r}. "
            f"Add it to PDF_REGISTRY or use load_pdf_generic()."
        )

    metadata = PDF_REGISTRY[filename]
    logger.info(f"Loading PDF: {filename} ({metadata['doc_id']})")

    text = _extract_text_from_pdf(pdf_path)

    return LoadedDocument(
        doc_id=metadata["doc_id"],
        title=metadata["title"],
        doc_type=metadata["doc_type"],
        source_org=metadata["source_org"],
        date_published=metadata["date_published"],
        is_normative=metadata["is_normative"],
        text=text,
        source_path=str(pdf_path),
        load_method="pdf_extract",
    )


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from PDF using PyMuPDF with layout preservation.

    Uses 'text' extraction mode which preserves reading order.
    For legal PDFs with two-column layouts (common in EUR-Lex),
    PyMuPDF handles column detection automatically.
    """
    doc = fitz.open(str(pdf_path))
    pages_text = []

    # EUR-Lex PDF footer pattern appearing mid-text between pages:
    # "EN\nOJ L, 12.7.2024\n56/144\nELI: http://data.europa.eu/..."
    footer_pattern = re.compile(
        r"EN\s*\nOJ L,\s*[\d.]+\n[\d/]+\nELI:\s*http\S*",
        re.DOTALL,
    )

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        page_text = footer_pattern.sub("", page_text)
        if page_text.strip():
            pages_text.append(f"[Page {page_num}]\n{page_text}")

    doc.close()

    full_text = "\n\n".join(pages_text)
    logger.info(
        f"Extracted {len(full_text):,} characters from "
        f"{len(pages_text)} pages of {pdf_path.name}"
    )
    return full_text


def load_pdf_generic(
    pdf_path: Path,
    doc_id: str,
    title: str,
    doc_type: str,
    source_org: str = "unknown",
    date_published: str = "unknown",
    is_normative: bool = False,
) -> LoadedDocument:
    """
    Load any PDF not in the registry — useful for adding new documents
    without modifying PDF_REGISTRY (e.g. national implementation acts in V2).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = _extract_text_from_pdf(pdf_path)

    return LoadedDocument(
        doc_id=doc_id,
        title=title,
        doc_type=doc_type,
        source_org=source_org,
        date_published=date_published,
        is_normative=is_normative,
        text=text,
        source_path=str(pdf_path),
        load_method="pdf_extract",
    )


# ---------------------------------------------------------------------------
# Corpus loader — loads all sources in one call
# ---------------------------------------------------------------------------

def load_corpus(
    data_raw_dir: Path,
    fetch_html: bool = False,  # EUR-Lex blocks automated requests (AWS WAF)
    skip_missing: bool = False,
) -> list[LoadedDocument]:
    """
    Load the complete AI Act Navigator corpus.

    Args:
        data_raw_dir: path to data/raw/ directory
        fetch_html:   EUR-Lex now blocks automated requests via AWS WAF.
                      Keep False (default) — PDF is the reliable source.
                      Set True only if EUR-Lex WAF protection is lifted.
        skip_missing: if True, warn on missing files instead of raising

    Returns:
        list of LoadedDocument, one per source
    """
    documents = []

    # 1. AI Act — HTML preferred, PDF fallback
    if fetch_html:
        try:
            doc = load_ai_act_html()
            documents.append(doc)
            logger.info(f"✓ AI Act HTML — {doc.word_count:,} words")
        except Exception as e:
            logger.warning(f"HTML fetch failed ({e}), falling back to PDF")
            fetch_html = False

    if not fetch_html:
        pdf_path = data_raw_dir / "ai_act_2024_1689.pdf"
        if pdf_path.exists():
            doc = load_pdf(pdf_path)
            documents.append(doc)
            logger.info(f"✓ AI Act PDF — {doc.word_count:,} words")
        else:
            raise FileNotFoundError(
                "AI Act PDF not found and HTML fetch failed.\n"
                "Download ai_act_2024_1689.pdf to data/raw/"
            )

    # 2. GPAI Code of Practice — 3 chapters
    for filename in [
        "gpai_cop_chapter1_transparency.pdf",
        "gpai_cop_chapter2_copyright.pdf",
        "gpai_cop_chapter3_safety_security.pdf",
    ]:
        pdf_path = data_raw_dir / filename
        if not pdf_path.exists():
            msg = f"Missing GPAI CoP file: {filename}"
            if skip_missing:
                logger.warning(f"⚠  {msg} — skipping")
                continue
            raise FileNotFoundError(msg)
        doc = load_pdf(pdf_path)
        documents.append(doc)
        logger.info(f"✓ {doc.doc_id} — {doc.word_count:,} words")

    # 3. GPAI Guidelines
    guidelines_path = data_raw_dir / "gpai_guidelines_commission_2025.pdf"
    if not guidelines_path.exists():
        msg = "Missing: gpai_guidelines_commission_2025.pdf"
        if skip_missing:
            logger.warning(f"⚠  {msg} — skipping")
        else:
            raise FileNotFoundError(msg)
    else:
        doc = load_pdf(guidelines_path)
        documents.append(doc)
        logger.info(f"✓ {doc.doc_id} — {doc.word_count:,} words")

    total_words = sum(d.word_count for d in documents)
    logger.info(
        f"\nCorpus loaded: {len(documents)} documents | "
        f"{total_words:,} total words"
    )
    return documents


# ---------------------------------------------------------------------------
# CLI — quick sanity check: python -m src.ingestion.loader
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data_raw = Path(__file__).parents[3] / "data" / "raw"

    print(f"\nLooking for source files in: {data_raw}\n")
    print("PDFs found:")
    for f in sorted(data_raw.glob("*.pdf")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:55s} {size_kb:>7.0f} KB")

    print("\nLoading corpus (skip_missing=True)...")
    docs = load_corpus(data_raw, fetch_html=True, skip_missing=True)

    print("\n=== Corpus summary ===")
    for doc in docs:
        print(f"  {doc}")

    total = sum(d.word_count for d in docs)
    print(f"\nTotal words across corpus: {total:,}")

    if docs:
        sample = docs[0]
        print(f"\n=== First 500 chars of '{sample.doc_id}' ===")
        print(sample.text[:500])
