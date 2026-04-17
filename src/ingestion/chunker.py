"""
chunker.py — Hierarchical document chunker for AI Act Navigator

Implements a 3-level chunking schema designed for legal text:

  Level 0 — Document section (Title / Chapter / Annex)
             Not stored as a chunk — injected as context header only

  Level 1 — Article or Annex section  (primary retrieval unit)
             Full article text, ~200–800 tokens
             Used when: broad obligation queries

  Level 2 — Paragraph / point         (fine-grained retrieval unit)
             Single numbered paragraph or lettered point, ~50–200 tokens
             Used when: precise legal reference needed

Every chunk carries:
  - A context header (prepended before embedding, stripped from display)
  - Rich metadata for Qdrant payload filtering

Observed PDF structure (from PyMuPDF extraction):
    CHAPTER III
    HIGH-RISK AI SYSTEMS
    SECTION 1
    Classification of AI systems as high-risk
    Article 6
    Classification rules for high-risk AI systems
    1. Irrespective of whether an AI system...
    2. In addition to the high-risk AI systems...
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .loader import LoadedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target token counts (approximate — 1 token ≈ 0.75 words for legal English)
LEVEL1_MAX_TOKENS = 800    # full article — split if longer
LEVEL2_TARGET_TOKENS = 200  # single paragraph
LEVEL2_MAX_TOKENS = 400    # hard cap before forced split

# Application deadlines — used to populate date_application metadata
DEADLINE_MAP = {
    "prohibited":     "2025-02-02",  # Art. 5 — already in force
    "gpai":           "2025-08-02",  # Art. 51–56 — already in force
    "high_risk":      "2026-08-02",  # Annex III systems
    "high_risk_products": "2027-08-02",  # Annex II Section B
    "default":        "2026-08-02",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single chunk ready for embedding and Qdrant indexing.

    text_to_embed:  context_header + text  (what gets embedded)
    text:           raw chunk text only     (what gets displayed)
    context_header: hierarchical position   (injected before embedding)
    """
    # Identity
    chunk_id: str
    doc_id: str
    level: int                    # 1 = article, 2 = paragraph

    # Content
    text: str                     # display text
    context_header: str           # prepended before embedding
    text_to_embed: str = field(init=False)

    # Legal position metadata
    title_num: Optional[str] = None
    title_name: Optional[str] = None
    chapter_num: Optional[str] = None
    chapter_name: Optional[str] = None
    section_num: Optional[str] = None
    section_name: Optional[str] = None
    article_num: Optional[str] = None
    article_name: Optional[str] = None
    paragraph_num: Optional[str] = None
    annex_id: Optional[str] = None
    annex_name: Optional[str] = None

    # Retrieval metadata (Qdrant payload filters)
    doc_type: str = "regulation"
    is_normative: bool = True
    is_recital: bool = False
    risk_tier_relevance: list[str] = field(default_factory=list)
    actor_relevance: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)
    date_application: Optional[str] = None
    source_org: str = "EUR-Lex"
    date_published: str = "2024-07-12"

    # Stats
    word_count: int = field(init=False)

    def __post_init__(self):
        self.text_to_embed = f"{self.context_header}\n\n{self.text}".strip()
        self.word_count = len(self.text.split())

    def to_dict(self) -> dict:
        """Serialize to dict for Qdrant payload."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "level": self.level,
            "text": self.text,
            "context_header": self.context_header,
            "title_num": self.title_num,
            "title_name": self.title_name,
            "chapter_num": self.chapter_num,
            "chapter_name": self.chapter_name,
            "section_num": self.section_num,
            "section_name": self.section_name,
            "article_num": self.article_num,
            "article_name": self.article_name,
            "paragraph_num": self.paragraph_num,
            "annex_id": self.annex_id,
            "annex_name": self.annex_name,
            "doc_type": self.doc_type,
            "is_normative": self.is_normative,
            "is_recital": self.is_recital,
            "risk_tier_relevance": self.risk_tier_relevance,
            "actor_relevance": self.actor_relevance,
            "cross_references": self.cross_references,
            "date_application": self.date_application,
            "source_org": self.source_org,
            "date_published": self.date_published,
            "word_count": self.word_count,
        }

    def __repr__(self) -> str:
        loc = self.article_num or self.annex_id or "?"
        return (
            f"Chunk(id={self.chunk_id!r}, "
            f"level={self.level}, "
            f"loc={loc!r}, "
            f"words={self.word_count})"
        )


# ---------------------------------------------------------------------------
# Regex patterns — calibrated to PyMuPDF output from EUR-Lex PDF
# ---------------------------------------------------------------------------

# Matches: "Article 6" or "Article 50a"
RE_ARTICLE = re.compile(
    r"^Article\s+(\d+[a-z]?)\s*$",
    re.MULTILINE,
)

# Matches: "CHAPTER III" or "CHAPTER 2"
RE_CHAPTER = re.compile(
    r"^CHAPTER\s+([IVXLCDM]+|\d+)\s*$",
    re.MULTILINE,
)

# Matches: "TITLE II" or "TITLE 3"
RE_TITLE = re.compile(
    r"^TITLE\s+([IVXLCDM]+|\d+)\s*$",
    re.MULTILINE,
)

# Matches: "SECTION 1" or "SECTION 2"
RE_SECTION = re.compile(
    r"^SECTION\s+(\d+)\s*$",
    re.MULTILINE,
)

# Matches annex headers: "ANNEX I", "ANNEX III", "ANNEX: ..."
RE_ANNEX = re.compile(
    r"^ANNEX(?::\s*|[\s]+)([IVXLCDM]+|[A-Z]+)?\s*$",
    re.MULTILINE,
)

# Matches paragraph numbers: "1.", "2.", "42." at line start
RE_PARAGRAPH = re.compile(
    r"^(\d+)\.\s+",
    re.MULTILINE,
)

# Matches lettered points: "(a)", "(b)", "(i)" at line start
RE_POINT = re.compile(
    r"^\(([a-z]|x{0,3}(?:ix|iv|v?i{0,3}))\)\s+",
    re.MULTILINE,
)

# Cross-reference detection
RE_CROSS_REF = re.compile(
    r"[Aa]rticle\s+(\d+[a-z]?)(?:\s*\(\d+\))?|[Aa]nnex\s+([IVXLCDM]+)",
)

# Roman numeral converter for sorting
ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13,
}


# ---------------------------------------------------------------------------
# Main chunker class
# ---------------------------------------------------------------------------

class AIActChunker:
    """
    Hierarchical chunker for AI Act and related GPAI documents.

    Routing logic:
    - AI Act (regulation) → _chunk_regulation()
      Parses Titles / Chapters / Articles / Paragraphs
    - GPAI CoP + Guidelines (PDF) → _chunk_pdf_document()
      Simpler section-based chunking since structure is less rigid
    """

    def __init__(self):
        self._chunk_counter: dict[str, int] = {}
        self._used_ids: set[str] = set()

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        """
        Dispatch to the correct chunking strategy based on doc_type.
        """
        logger.info(f"Chunking {doc.doc_id!r} ({doc.doc_type})")

        if doc.doc_type == "regulation":
            chunks = self._chunk_regulation(doc)
        else:
            chunks = self._chunk_pdf_document(doc)

        logger.info(
            f"  → {len(chunks)} chunks "
            f"(avg {sum(c.word_count for c in chunks) // max(len(chunks),1)} words)"
        )
        return chunks

    def chunk_corpus(self, documents: list[LoadedDocument]) -> list[Chunk]:
        """Chunk all documents in the corpus."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"\nTotal chunks: {len(all_chunks)} across "
            f"{len(documents)} documents"
        )
        return all_chunks

    # -----------------------------------------------------------------------
    # Regulation chunker (AI Act)
    # -----------------------------------------------------------------------

    def _chunk_regulation(self, doc: LoadedDocument) -> list[Chunk]:
        """
        Parse the AI Act into hierarchical chunks.

        Strategy:
        1. Split text at Article boundaries → one block per article
        2. For each article block, extract paragraph-level Level-2 chunks
        3. Also store the full article as a Level-1 chunk
        4. Detect and separately chunk Annex sections
        """
        chunks = []
        text = doc.text

        # --- Split into article blocks ---
        # Find all Article positions
        article_matches = list(RE_ARTICLE.finditer(text))

        if not article_matches:
            logger.warning(
                f"No articles found in {doc.doc_id} — "
                f"falling back to generic chunking"
            )
            return self._chunk_pdf_document(doc)

        logger.info(f"  Found {len(article_matches)} articles")

        # Build context tracker — walk through text to track
        # current Title / Chapter / Section as we encounter them
        context = _ContextTracker(text)

        for i, match in enumerate(article_matches):
            art_num = match.group(1)
            art_start = match.start()
            art_end = (
                article_matches[i + 1].start()
                if i + 1 < len(article_matches)
                else len(text)
            )

            # Text from "Article X\n" to next article
            art_block = text[art_start:art_end].strip()
            lines = art_block.splitlines()

            # Article title is the line immediately after "Article N"
            art_title = ""
            body_start_line = 1
            if len(lines) > 1 and lines[1].strip() and not RE_PARAGRAPH.match(lines[1]):
                art_title = lines[1].strip()
                body_start_line = 2

            body = "\n".join(lines[body_start_line:]).strip()

            # Update context (Title/Chapter/Section) from text before this article
            ctx = context.get_context_at(art_start)

            # --- Level 1: full article chunk ---
            article_context_header = _build_article_header(
                ctx, art_num, art_title
            )
            l1_chunk = self._make_chunk(
                doc=doc,
                level=1,
                text=art_block,
                context_header=article_context_header,
                article_num=art_num,
                article_name=art_title,
                **ctx,
            )
            chunks.append(l1_chunk)

            # --- Level 2: paragraph chunks ---
            para_chunks = self._split_into_paragraphs(
                doc=doc,
                body=body,
                article_num=art_num,
                article_name=art_title,
                context_header_base=article_context_header,
                ctx=ctx,
            )
            chunks.extend(para_chunks)

        # --- Annex chunks ---
        annex_chunks = self._chunk_annexes(doc, text)
        chunks.extend(annex_chunks)

        return chunks

    def _split_into_paragraphs(
        self,
        doc: LoadedDocument,
        body: str,
        article_num: str,
        article_name: str,
        context_header_base: str,
        ctx: dict,
    ) -> list[Chunk]:
        """
        Split article body into paragraph-level Level-2 chunks.

        Paragraph boundaries are numbered lines: "1. ...", "2. ..."
        Points "(a)", "(b)" are kept with their parent paragraph.
        """
        chunks = []

        # Find paragraph boundaries
        para_matches = list(RE_PARAGRAPH.finditer(body))

        if not para_matches:
            # No numbered paragraphs — article is a single block
            # Only create L2 if meaningfully different from L1
            return []

        for j, pmatch in enumerate(para_matches):
            para_num = pmatch.group(1)
            para_start = pmatch.start()
            para_end = (
                para_matches[j + 1].start()
                if j + 1 < len(para_matches)
                else len(body)
            )
            para_text = body[para_start:para_end].strip()

            if len(para_text.split()) < 10:
                continue  # skip near-empty paragraphs

            para_header = (
                f"{context_header_base}\n"
                f"[Article {article_num}, paragraph {para_num}]"
            )

            chunk = self._make_chunk(
                doc=doc,
                level=2,
                text=para_text,
                context_header=para_header,
                article_num=article_num,
                article_name=article_name,
                paragraph_num=para_num,
                **ctx,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_annexes(
        self,
        doc: LoadedDocument,
        text: str,
    ) -> list[Chunk]:
        """
        Chunk the Annexes section of the AI Act.

        Annex structure varies significantly:
        - Annex I: flat list of AI techniques
        - Annex III: hierarchical numbered list of high-risk domains
        - Annex IV: documentation requirements
        Each Annex is stored as a Level-1 chunk.
        Sub-sections within Annex III are stored as Level-2 chunks.
        """
        chunks = []
        annex_matches = list(RE_ANNEX.finditer(text))

        if not annex_matches:
            logger.warning("No annexes found in regulation text")
            return []

        logger.info(f"  Found {len(annex_matches)} annexes")

        for i, match in enumerate(annex_matches):
            annex_id_raw = (match.group(1) or "").strip()
            annex_start = match.start()
            annex_end = (
                annex_matches[i + 1].start()
                if i + 1 < len(annex_matches)
                else len(text)
            )

            annex_block = text[annex_start:annex_end].strip()
            lines = annex_block.splitlines()

            # Annex title is 1-2 lines after the ANNEX header
            annex_title = ""
            if len(lines) > 1:
                annex_title = lines[1].strip()

            annex_id = f"annex_{annex_id_raw.lower()}" if annex_id_raw else f"annex_{i+1}"

            header = f"[Annex {annex_id_raw}]\n[{annex_title}]"

            chunk = self._make_chunk(
                doc=doc,
                level=1,
                text=annex_block,
                context_header=header,
                annex_id=annex_id,
                annex_name=annex_title,
            )
            # Annex-specific metadata
            chunk.risk_tier_relevance = _annex_risk_tiers(annex_id_raw)
            chunk.date_application = _annex_deadline(annex_id_raw)
            chunks.append(chunk)

        return chunks

    # -----------------------------------------------------------------------
    # Generic PDF chunker (GPAI CoP + Guidelines)
    # -----------------------------------------------------------------------

    def _chunk_pdf_document(self, doc: LoadedDocument) -> list[Chunk]:
        """
        Chunk GPAI documents by page blocks and paragraph breaks.

        GPAI documents don't have the rigid Article/Paragraph structure
        of the AI Act, so we use a simpler sliding-window approach with
        natural paragraph boundaries as split points.
        """
        chunks = []
        text = doc.text

        # Split on [Page N] markers inserted by loader
        page_blocks = re.split(r"\[Page \d+\]", text)
        page_blocks = [b.strip() for b in page_blocks if b.strip()]

        # Group pages into chunks of ~LEVEL2_TARGET_TOKENS words
        current_chunk_lines = []
        current_word_count = 0
        chunk_index = 0

        for block in page_blocks:
            block_words = len(block.split())

            if (current_word_count + block_words > LEVEL2_MAX_TOKENS
                    and current_chunk_lines):
                # Flush current chunk
                chunk_text = "\n\n".join(current_chunk_lines)
                header = (
                    f"[{doc.title}]\n"
                    f"[Section {chunk_index + 1}]"
                )
                chunk = self._make_chunk(
                    doc=doc,
                    level=2,
                    text=chunk_text,
                    context_header=header,
                )
                chunks.append(chunk)
                chunk_index += 1
                current_chunk_lines = []
                current_word_count = 0

            current_chunk_lines.append(block)
            current_word_count += block_words

        # Flush remaining content
        if current_chunk_lines:
            chunk_text = "\n\n".join(current_chunk_lines)
            header = (
                f"[{doc.title}]\n"
                f"[Section {chunk_index + 1}]"
            )
            chunk = self._make_chunk(
                doc=doc,
                level=2,
                text=chunk_text,
                context_header=header,
            )
            chunks.append(chunk)

        return chunks

    # -----------------------------------------------------------------------
    # Chunk factory
    # -----------------------------------------------------------------------

    def _make_chunk(
        self,
        doc: LoadedDocument,
        level: int,
        text: str,
        context_header: str,
        article_num: Optional[str] = None,
        article_name: Optional[str] = None,
        paragraph_num: Optional[str] = None,
        annex_id: Optional[str] = None,
        annex_name: Optional[str] = None,
        title_num: Optional[str] = None,
        title_name: Optional[str] = None,
        chapter_num: Optional[str] = None,
        chapter_name: Optional[str] = None,
        section_num: Optional[str] = None,
        section_name: Optional[str] = None,
    ) -> Chunk:
        """Create a Chunk with auto-generated ID and enriched metadata."""

        # Generate unique chunk ID
        key = doc.doc_id
        self._chunk_counter[key] = self._chunk_counter.get(key, 0) + 1
        n = self._chunk_counter[key]

        if article_num:
            chunk_id = f"{doc.doc_id}_art{article_num}"
            if paragraph_num:
                chunk_id += f"_p{paragraph_num}"
        elif annex_id:
            chunk_id = f"{doc.doc_id}_{annex_id}"
        else:
            chunk_id = f"{doc.doc_id}_chunk{n:04d}"

        # Guarantee uniqueness — append suffix if ID already used
        # (happens when PDF repeats article headers across page breaks)
        if chunk_id in self._used_ids:
            suffix = 2
            candidate = f"{chunk_id}_v{suffix}"
            while candidate in self._used_ids:
                suffix += 1
                candidate = f"{chunk_id}_v{suffix}"
            logger.debug(
                f"Duplicate chunk_id resolved: {chunk_id!r} → {candidate!r}"
            )
            chunk_id = candidate
        self._used_ids.add(chunk_id)

        # Extract cross-references from text
        refs = _extract_cross_references(text)

        # Infer risk tier and actor relevance from article number
        risk_tiers = _infer_risk_tiers(article_num, annex_id)
        actors = _infer_actors(article_num, text)
        deadline = _infer_deadline(article_num, annex_id, risk_tiers)

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            level=level,
            text=text,
            context_header=context_header,
            title_num=title_num,
            title_name=title_name,
            chapter_num=chapter_num,
            chapter_name=chapter_name,
            section_num=section_num,
            section_name=section_name,
            article_num=article_num,
            article_name=article_name,
            paragraph_num=paragraph_num,
            annex_id=annex_id,
            annex_name=annex_name,
            doc_type=doc.doc_type,
            is_normative=doc.is_normative,
            cross_references=refs,
            risk_tier_relevance=risk_tiers,
            actor_relevance=actors,
            date_application=deadline,
            source_org=doc.source_org,
            date_published=doc.date_published,
        )


# ---------------------------------------------------------------------------
# Context tracker — walks text to maintain Title/Chapter/Section state
# ---------------------------------------------------------------------------

class _ContextTracker:
    """
    Scans the document text once and builds a position→context index.
    Used to know which Title/Chapter/Section we're in at any text offset.
    """

    def __init__(self, text: str):
        self._events: list[tuple[int, str, str]] = []  # (pos, key, value)
        self._index_text(text)

    def _index_text(self, text: str):
        # Structural keywords that should never be captured as names
        _skip = re.compile(
            r"^(TITLE|CHAPTER|SECTION|ANNEX|Article)\s",
            re.IGNORECASE,
        )

        def _find_name(after: str) -> Optional[str]:
            """Find first non-empty, non-structural line after a header."""
            for line in after.splitlines():
                line = line.strip()
                if line and not _skip.match(line):
                    return line
            return None

        for m in RE_TITLE.finditer(text):
            self._events.append((m.start(), "title_num", m.group(1)))
            # Reset chapter/section context on new title
            self._events.append((m.start() + 1, "chapter_num", None))
            self._events.append((m.start() + 1, "chapter_name", None))
            self._events.append((m.start() + 1, "section_num", None))
            self._events.append((m.start() + 1, "section_name", None))
            after = text[m.end():m.end() + 300]
            name = _find_name(after)
            if name:
                self._events.append((m.start() + 1, "title_name", name))

        for m in RE_CHAPTER.finditer(text):
            self._events.append((m.start(), "chapter_num", m.group(1)))
            # Reset section context on new chapter
            self._events.append((m.start() + 1, "section_num", None))
            self._events.append((m.start() + 1, "section_name", None))
            after = text[m.end():m.end() + 300]
            name = _find_name(after)
            if name:
                self._events.append((m.start() + 1, "chapter_name", name))

        for m in RE_SECTION.finditer(text):
            self._events.append((m.start(), "section_num", m.group(1)))
            after = text[m.end():m.end() + 300]
            name = _find_name(after)
            if name:
                self._events.append((m.start() + 1, "section_name", name))

        self._events.sort(key=lambda x: x[0])

    def get_context_at(self, position: int) -> dict:
        """Return the active Title/Chapter/Section context at a text position."""
        ctx: dict[str, Optional[str]] = {
            "title_num": None, "title_name": None,
            "chapter_num": None, "chapter_name": None,
            "section_num": None, "section_name": None,
        }
        for pos, key, value in self._events:
            if pos <= position:
                ctx[key] = value
            else:
                break
        return ctx


# ---------------------------------------------------------------------------
# Metadata inference helpers
# ---------------------------------------------------------------------------

def _build_article_header(ctx: dict, art_num: str, art_title: str) -> str:
    parts = []
    if ctx.get("title_num"):
        title_part = f"Title {ctx['title_num']}"
        if ctx.get("title_name"):
            title_part += f" — {ctx['title_name']}"
        parts.append(title_part)
    if ctx.get("chapter_num"):
        chap_part = f"Chapter {ctx['chapter_num']}"
        if ctx.get("chapter_name"):
            chap_part += f" — {ctx['chapter_name']}"
        parts.append(chap_part)
    if ctx.get("section_num"):
        sec_part = f"Section {ctx['section_num']}"
        if ctx.get("section_name"):
            sec_part += f" — {ctx['section_name']}"
        parts.append(sec_part)
    parts.append(f"Article {art_num} — {art_title}")
    return "[" + "]\n[".join(parts) + "]"


def _extract_cross_references(text: str) -> list[str]:
    refs = set()
    for m in RE_CROSS_REF.finditer(text):
        if m.group(1):
            refs.add(f"art_{m.group(1)}")
        if m.group(2):
            refs.add(f"annex_{m.group(2).lower()}")
    return sorted(refs)


def _infer_risk_tiers(
    article_num: Optional[str],
    annex_id: Optional[str],
) -> list[str]:
    """Infer which risk tiers this chunk is relevant to."""
    tiers = []
    if not article_num:
        if annex_id in ("annex_iii",):
            return ["high_risk"]
        if annex_id in ("annex_i",):
            return ["high_risk", "limited_risk", "minimal_risk"]
        return []

    num = int(re.sub(r"[a-z]", "", article_num))

    if num == 5:
        tiers = ["prohibited"]
    elif num in (6, 7) or (8 <= num <= 49):
        tiers = ["high_risk"]
    elif num == 50:
        tiers = ["limited_risk"]
    elif 51 <= num <= 56:
        tiers = ["gpai"]
    else:
        tiers = ["all"]

    return tiers


def _infer_actors(
    article_num: Optional[str],
    text: str,
) -> list[str]:
    """Infer which actors (provider/deployer) this chunk addresses."""
    actors = []
    text_lower = text.lower()
    if "provider" in text_lower:
        actors.append("provider")
    if "deployer" in text_lower:
        actors.append("deployer")
    if "importer" in text_lower:
        actors.append("importer")
    if "distributor" in text_lower:
        actors.append("distributor")
    return actors or ["all"]


def _infer_deadline(
    article_num: Optional[str],
    annex_id: Optional[str],
    risk_tiers: list[str],
) -> str:
    """Map chunk to its applicable compliance deadline."""
    if "prohibited" in risk_tiers:
        return DEADLINE_MAP["prohibited"]
    if "gpai" in risk_tiers:
        return DEADLINE_MAP["gpai"]
    if "high_risk" in risk_tiers:
        if annex_id == "annex_ix":   # products — later deadline
            return DEADLINE_MAP["high_risk_products"]
        return DEADLINE_MAP["high_risk"]
    return DEADLINE_MAP["default"]


def _annex_risk_tiers(annex_id_raw: str) -> list[str]:
    mapping = {
        "I": ["high_risk", "gpai"],
        "II": ["high_risk"],
        "III": ["high_risk"],
        "IV": ["high_risk"],
        "V": ["high_risk"],
        "VI": ["high_risk"],
        "VII": ["high_risk"],
        "VIII": ["high_risk"],
        "IX": ["high_risk"],
        "XI": ["gpai"],
        "XII": ["gpai"],
    }
    return mapping.get(annex_id_raw, ["all"])


def _annex_deadline(annex_id_raw: str) -> str:
    gpai_annexes = {"XI", "XII"}
    if annex_id_raw in gpai_annexes:
        return DEADLINE_MAP["gpai"]
    return DEADLINE_MAP["high_risk"]


# ---------------------------------------------------------------------------
# CLI — sanity check: python -m src.ingestion.chunker
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from .loader import load_corpus

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data_raw = Path(__file__).parents[3] / "data" / "raw"

    print("Loading corpus...")
    docs = load_corpus(data_raw, fetch_html=False, skip_missing=True)

    print("\nChunking corpus...")
    chunker = AIActChunker()
    chunks = chunker.chunk_corpus(docs)

    # Summary stats
    print(f"\n{'='*50}")
    print(f"Total chunks:     {len(chunks)}")
    print(f"Level 1 chunks:   {sum(1 for c in chunks if c.level == 1)}")
    print(f"Level 2 chunks:   {sum(1 for c in chunks if c.level == 2)}")

    avg_words = sum(c.word_count for c in chunks) // max(len(chunks), 1)
    print(f"Avg words/chunk:  {avg_words}")

    # Sample: show first 3 article chunks with their context headers
    print(f"\n{'='*50}")
    print("Sample Level-1 article chunks:")
    art_chunks = [c for c in chunks if c.level == 1 and c.article_num][:3]
    for c in art_chunks:
        print(f"\n  {c}")
        print(f"  Context header:\n    " +
              c.context_header.replace("\n", "\n    "))
        print(f"  Text preview: {c.text[:120].strip()}...")
        print(f"  Risk tiers: {c.risk_tier_relevance}")
        print(f"  Deadline: {c.date_application}")
        print(f"  Cross-refs: {c.cross_references[:5]}")

    # Sample: show a Level-2 paragraph chunk
    print(f"\n{'='*50}")
    print("Sample Level-2 paragraph chunk:")
    para_chunks = [c for c in chunks if c.level == 2 and c.paragraph_num][:1]
    for c in para_chunks:
        print(f"\n  {c}")
        print(f"  text_to_embed preview:\n    " +
              c.text_to_embed[:300].replace("\n", "\n    "))
