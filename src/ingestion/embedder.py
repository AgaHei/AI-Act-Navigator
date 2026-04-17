"""
embedder.py — Dense + sparse embedding for AI Act Navigator

Produces two vector representations per chunk:

  Dense vector  — mistral-embed (1024 dims)
                  Encodes semantic meaning of text_to_embed
                  (context header + chunk text)

  Sparse vector — BM25 (variable dims, dict of {token_id: score})
                  Encodes exact term frequency weighted by corpus rarity
                  Fit on the full corpus so IDF scores are meaningful

Both are stored together in an EmbeddedChunk, ready for Qdrant indexing.

Design decisions:
  - Batching: 50 chunks per Mistral API call (safe rate limit margin)
  - BM25 fit: on chunk.text only (not text_to_embed) — context headers
    would inflate IDF scores for structural terms like "Article", "Chapter"
  - Retry logic: exponential backoff on API errors
  - Progress: tqdm bar so you can monitor a ~5 min embedding run
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .chunker import Chunk

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MISTRAL_EMBED_MODEL = "mistral-embed"
EMBED_BATCH_SIZE = 50        # chunks per API call — safe for rate limits
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0       # seconds, doubles on each retry

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedChunk:
    """
    A chunk enriched with both dense and sparse vector representations.
    Ready for upsert into Qdrant.
    """
    chunk: Chunk
    dense_vector: list[float]         # 1024-dim mistral-embed output
    sparse_indices: list[int]         # BM25 non-zero token indices
    sparse_values: list[float]        # BM25 corresponding scores

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def dense_dim(self) -> int:
        return len(self.dense_vector)

    @property
    def sparse_nnz(self) -> int:
        """Number of non-zero sparse dimensions."""
        return len(self.sparse_indices)

    def __repr__(self) -> str:
        return (
            f"EmbeddedChunk(id={self.chunk_id!r}, "
            f"dense={self.dense_dim}d, "
            f"sparse_nnz={self.sparse_nnz})"
        )


# ---------------------------------------------------------------------------
# BM25 sparse encoder
# ---------------------------------------------------------------------------

class BM25Encoder:
    """
    Fits a BM25 model on the full corpus and encodes individual chunks
    as sparse vectors (dict of non-zero {token_index: score} pairs).

    Must be fit on ALL chunks before encoding any individual chunk,
    so that IDF scores reflect corpus-wide term rarity.
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._vocabulary: dict[str, int] = {}  # token → index
        self._fitted = False

    def fit(self, chunks: list[Chunk]) -> "BM25Encoder":
        """
        Fit BM25 on chunk texts (not text_to_embed — see module docstring).

        Args:
            chunks: all chunks in the corpus

        Returns:
            self (for chaining)
        """
        logger.info(f"Fitting BM25 on {len(chunks)} chunks...")

        tokenized = [self._tokenize(c.text) for c in chunks]

        # Initialize BM25 first
        self._bm25 = BM25Okapi(tokenized)
        
        # Build vocabulary by reconstructing BM25's token order
        # from the doc_freqs which contains term frequencies per document
        all_tokens = set()
        for doc_freqs in self._bm25.doc_freqs:
            all_tokens.update(doc_freqs.keys())
        
        # Sort tokens to get consistent ordering
        sorted_tokens = sorted(all_tokens)
        self._vocabulary = {token: idx for idx, token in enumerate(sorted_tokens)}
        
        self._fitted = True

        logger.info(f"BM25 fitted — vocabulary size: {len(self._vocabulary):,} tokens")
        return self

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """
        Encode a single text as a sparse vector using TF-IDF style scoring.

        Args:
            text: chunk text to encode

        Returns:
            (indices, values) — parallel lists of non-zero token positions
            and their scores. Empty if no known tokens found.
        """
        if not self._fitted:
            raise RuntimeError("BM25Encoder must be fit() before encode()")

        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        # Count token frequencies in this document
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        scores: dict[int, float] = {}
        doc_len = len(tokens)
        avgdl = self._bm25.avgdl
        k1 = self._bm25.k1
        b = self._bm25.b

        for token, count in token_counts.items():
            if token not in self._vocabulary:
                continue
                
            vocab_idx = self._vocabulary[token]
            
            # Calculate document frequency for this token across corpus
            doc_freq = sum(1 for doc_freqs in self._bm25.doc_freqs if token in doc_freqs)
            
            # Calculate IDF manually: log((N - df + 0.5) / (df + 0.5))
            N = self._bm25.corpus_size
            idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5))
            
            # Calculate BM25 term frequency component
            tf = count
            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            
            score = float(idf * tf_component)
            if score > 0:
                scores[vocab_idx] = score

        if not scores:
            return [], []

        indices = list(scores.keys())
        values = list(scores.values())
        return indices, values

    def encode_batch(
        self, chunks: list[Chunk]
    ) -> list[tuple[list[int], list[float]]]:
        """Encode multiple chunks efficiently."""
        return [self.encode(c.text) for c in chunks]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple whitespace + lowercase tokenizer.
        Legal text doesn't need stemming — exact term matching is the goal.
        Filters out pure punctuation and single characters.
        """
        import re
        tokens = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9\-]*\b", text.lower())
        return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# Dense embedder (Mistral)
# ---------------------------------------------------------------------------

class MistralEmbedder:
    """
    Calls mistral-embed API in batches with retry logic.

    Embeds chunk.text_to_embed (context header + text) so the dense
    vector captures the chunk's hierarchical position in the document.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise ValueError(
                "MISTRAL_API_KEY not found. "
                "Set it in .env or pass api_key= explicitly."
            )
        self._client = Mistral(api_key=key)
        self._model = MISTRAL_EMBED_MODEL

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts with retry on API errors.

        Args:
            texts: list of strings to embed (max EMBED_BATCH_SIZE)

        Returns:
            list of 1024-dim float vectors, one per input text
        """
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    inputs=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                last_error = e
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Embedding attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Mistral embedding failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> list[list[float]]:
        """
        Embed all chunks in batches with a progress bar.

        Args:
            chunks: all chunks to embed
            batch_size: number of chunks per API call

        Returns:
            list of dense vectors, same order as input chunks
        """
        all_vectors = []
        batches = [
            chunks[i: i + batch_size]
            for i in range(0, len(chunks), batch_size)
        ]

        logger.info(
            f"Embedding {len(chunks)} chunks in "
            f"{len(batches)} batches of {batch_size}..."
        )

        for batch in tqdm(batches, desc="Embedding", unit="batch"):
            texts = []
            for chunk in batch:
                text = self._truncate_if_needed(chunk.text_to_embed, chunk.chunk_id)
                texts.append(text)
            vectors = self.embed_batch(texts)
            all_vectors.extend(vectors)

            # Small delay between batches to stay within rate limits
            time.sleep(0.1)

        logger.info(f"Embedding complete — {len(all_vectors)} vectors produced")
        return all_vectors

    def _truncate_if_needed(self, text: str, chunk_id: str) -> str:
        """
        Truncate text if it exceeds Mistral's token limit (~8,192 tokens).
        Uses rough estimation: 1 token ≈ 0.75 words.
        """
        words = text.split()
        estimated_tokens = len(words) / 0.75
        
        if estimated_tokens <= 6000:  # Conservative limit
            return text
            
        # Truncate to ~5,500 tokens worth of words  
        max_words = int(5500 * 0.75)
        truncated_words = words[:max_words]
        truncated_text = " ".join(truncated_words) + "... [truncated]"
        
        logger.warning(
            f"Truncated {chunk_id}: {len(words)} words "
            f"(~{int(estimated_tokens)} tokens) → {len(truncated_words)} words"
        )
        return truncated_text


# ---------------------------------------------------------------------------
# Main embedding pipeline
# ---------------------------------------------------------------------------

def embed_corpus(
    chunks: list[Chunk],
    api_key: Optional[str] = None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    """
    Full embedding pipeline: fit BM25, then embed all chunks with Mistral.

    Args:
        chunks:     all chunks from the chunker (814 for our corpus)
        api_key:    Mistral API key (defaults to MISTRAL_API_KEY env var)
        batch_size: Mistral API batch size

    Returns:
        list of EmbeddedChunk, one per input chunk, same order
    """
    logger.info(f"Starting embedding pipeline for {len(chunks)} chunks")

    # Step 1 — Fit BM25 on full corpus (must come before encoding)
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(chunks)

    # Step 2 — Dense embeddings via Mistral
    mistral_embedder = MistralEmbedder(api_key=api_key)
    dense_vectors = mistral_embedder.embed_chunks(chunks, batch_size=batch_size)

    # Step 3 — Sparse vectors via BM25
    logger.info("Computing BM25 sparse vectors...")
    sparse_results = bm25_encoder.encode_batch(chunks)

    # Step 4 — Zip everything together
    embedded_chunks = []
    for chunk, dense_vec, (sparse_idx, sparse_vals) in zip(
        chunks, dense_vectors, sparse_results
    ):
        embedded_chunks.append(
            EmbeddedChunk(
                chunk=chunk,
                dense_vector=dense_vec,
                sparse_indices=sparse_idx,
                sparse_values=sparse_vals,
            )
        )

    # Sanity check
    assert len(embedded_chunks) == len(chunks), "Embedding count mismatch"

    sparse_nnz_avg = sum(e.sparse_nnz for e in embedded_chunks) // len(embedded_chunks)
    logger.info(
        f"Pipeline complete:\n"
        f"  {len(embedded_chunks)} embedded chunks\n"
        f"  Dense: {embedded_chunks[0].dense_dim} dims\n"
        f"  Sparse: avg {sparse_nnz_avg} non-zero dims per chunk"
    )

    return embedded_chunks


# ---------------------------------------------------------------------------
# Save / load — avoid re-embedding after chunker fixes
# ---------------------------------------------------------------------------

def save_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
    path: Path,
) -> None:
    """
    Persist embedded chunks to disk (vectors + chunk metadata).
    Allows re-indexing without re-calling the Mistral API.

    Format: JSON lines — one EmbeddedChunk per line for streaming load.
    """
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in embedded_chunks:
            record = {
                "chunk": e.chunk.to_dict(),
                "dense_vector": e.dense_vector,
                "sparse_indices": e.sparse_indices,
                "sparse_values": e.sparse_values,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / 1024 / 1024
    logger.info(f"Saved {len(embedded_chunks)} embedded chunks to {path} ({size_mb:.1f} MB)")


def load_embedded_chunks(path: Path) -> list[EmbeddedChunk]:
    """
    Load embedded chunks from disk, replacing chunk IDs with
    freshly re-chunked IDs if needed.

    Use this after fixing the chunker to re-index without re-embedding:

        # 1. Re-chunk to get correct IDs
        chunks = chunker.chunk_corpus(docs)

        # 2. Load saved vectors
        embedded = load_embedded_chunks(Path("data/embeddings/embedded_chunks.jsonl"))

        # 3. Patch chunk IDs from fresh chunks (vectors stay the same)
        embedded = patch_chunk_ids(embedded, chunks)

        # 4. Re-index
        index_corpus(embedded, recreate=True)
    """
    import json
    if not path.exists():
        raise FileNotFoundError(f"Embedded chunks file not found: {path}")

    embedded_chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunk_data = record["chunk"]

            # Reconstruct Chunk from dict
            chunk = Chunk(
                chunk_id=chunk_data["chunk_id"],
                doc_id=chunk_data["doc_id"],
                level=chunk_data["level"],
                text=chunk_data["text"],
                context_header=chunk_data["context_header"],
                title_num=chunk_data.get("title_num"),
                title_name=chunk_data.get("title_name"),
                chapter_num=chunk_data.get("chapter_num"),
                chapter_name=chunk_data.get("chapter_name"),
                section_num=chunk_data.get("section_num"),
                section_name=chunk_data.get("section_name"),
                article_num=chunk_data.get("article_num"),
                article_name=chunk_data.get("article_name"),
                paragraph_num=chunk_data.get("paragraph_num"),
                annex_id=chunk_data.get("annex_id"),
                annex_name=chunk_data.get("annex_name"),
                doc_type=chunk_data.get("doc_type", "regulation"),
                is_normative=chunk_data.get("is_normative", True),
                is_recital=chunk_data.get("is_recital", False),
                risk_tier_relevance=chunk_data.get("risk_tier_relevance", []),
                actor_relevance=chunk_data.get("actor_relevance", []),
                cross_references=chunk_data.get("cross_references", []),
                date_application=chunk_data.get("date_application"),
                source_org=chunk_data.get("source_org", "EUR-Lex"),
                date_published=chunk_data.get("date_published", "2024-07-12"),
            )
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    dense_vector=record["dense_vector"],
                    sparse_indices=record["sparse_indices"],
                    sparse_values=record["sparse_values"],
                )
            )

    logger.info(f"Loaded {len(embedded_chunks)} embedded chunks from {path}")
    return embedded_chunks


def patch_chunk_ids(
    embedded_chunks: list[EmbeddedChunk],
    fresh_chunks: list[Chunk],
) -> list[EmbeddedChunk]:
    """
    Replace chunk IDs in embedded chunks with IDs from freshly re-chunked data.

    Matches by position (index) — assumes chunk order is stable across
    chunker runs on the same corpus, which it is for deterministic chunking.

    Args:
        embedded_chunks: loaded from disk (old IDs)
        fresh_chunks:    output of chunker.chunk_corpus() (new unique IDs)

    Returns:
        embedded_chunks with updated chunk_ids
    """
    if len(embedded_chunks) != len(fresh_chunks):
        raise ValueError(
            f"Cannot patch IDs: embedded={len(embedded_chunks)} chunks "
            f"but fresh={len(fresh_chunks)} chunks. Counts must match."
        )

    patched = []
    for embedded, fresh in zip(embedded_chunks, fresh_chunks):
        # Replace the chunk entirely with the fresh one,
        # keeping the existing vectors
        patched.append(
            EmbeddedChunk(
                chunk=fresh,
                dense_vector=embedded.dense_vector,
                sparse_indices=embedded.sparse_indices,
                sparse_values=embedded.sparse_values,
            )
        )

    logger.info(f"Patched {len(patched)} chunk IDs from fresh chunker output")
    return patched


# ---------------------------------------------------------------------------
# Cost estimator — call before running to confirm budget
# ---------------------------------------------------------------------------

def estimate_cost(chunks: list[Chunk]) -> None:
    """
    Print a cost estimate before running the full embedding.
    mistral-embed: $0.10 per 1M tokens (approx 0.75 words per token)
    """
    total_words = sum(len(c.text_to_embed.split()) for c in chunks)
    estimated_tokens = int(total_words / 0.75)
    estimated_cost_usd = (estimated_tokens / 1_000_000) * 0.10

    print(f"\n{'='*45}")
    print(f"Embedding cost estimate")
    print(f"{'='*45}")
    print(f"  Chunks to embed:     {len(chunks):,}")
    print(f"  Total words:         {total_words:,}")
    print(f"  Estimated tokens:    {estimated_tokens:,}")
    print(f"  Estimated cost:      ${estimated_cost_usd:.4f} USD")
    print(f"{'='*45}\n")


# ---------------------------------------------------------------------------
# CLI — python -m src.ingestion.embedder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path
    from .loader import load_corpus
    from .chunker import AIActChunker

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data_raw = Path(__file__).parents[2] / "data" / "raw"
    data_processed = Path(__file__).parents[2] / "data" / "processed"

    # Load and chunk
    print("Loading corpus...")
    docs = load_corpus(data_raw, fetch_html=False, skip_missing=True)

    print("Chunking...")
    chunker = AIActChunker()
    chunks = chunker.chunk_corpus(docs)

    # Show cost estimate and confirm
    estimate_cost(chunks)
    confirm = input("Proceed with embedding? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        exit(0)

    # Embed
    embedded = embed_corpus(chunks)

    # Save full vectors to disk — enables re-indexing without re-embedding
    embeddings_dir = Path(__file__).parents[2] / "data" / "embeddings"
    vectors_file = embeddings_dir / "embedded_chunks.jsonl"
    save_embedded_chunks(embedded, vectors_file)
    print(f"✅ Full vectors saved to {vectors_file}")

    # Save lightweight metadata for inspection
    print("\nSaving metadata summary...")
    output = []
    for e in embedded:
        output.append({
            "chunk_id": e.chunk_id,
            "doc_id": e.chunk.doc_id,
            "dense_dim": e.dense_dim,
            "sparse_nnz": e.sparse_nnz,
            "dense_vector_sample": e.dense_vector[:5],
            "sparse_sample": dict(zip(
                e.sparse_indices[:5], e.sparse_values[:5]
            )),
        })

    out_file = data_processed / "embedded_chunks_meta.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Metadata saved to {out_file}")
    print(f"\nSample embedded chunk:")
    print(f"  {embedded[0]}")
    print(f"  Dense sample (first 5 dims): {embedded[0].dense_vector[:5]}")
    print(f"  Sparse nnz: {embedded[0].sparse_nnz}")
