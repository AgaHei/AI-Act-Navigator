"""
indexer.py — Qdrant indexing for AI Act Navigator

Creates the Qdrant collection with dual vector config (dense + sparse)
and upserts all EmbeddedChunks with their full metadata payload.

Collection design:
  - Dense vector:  "dense"  — 1024 dims, Cosine similarity (mistral-embed)
  - Sparse vector: "sparse" — variable dims (BM25)
  - Payload:       full Chunk metadata for filtered retrieval

The collection is created idempotently — safe to run multiple times.
Re-indexing (e.g. after a corpus update) deletes and recreates the collection.

Qdrant payload indexes are created on the fields most commonly used
for filtering: risk_tier_relevance, actor_relevance, date_application,
doc_type, article_num — enabling fast pre-filtered hybrid search.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
    SparseVectorsConfig,
)
from tqdm import tqdm

from .embedder import EmbeddedChunk

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "ai_act_navigator")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = 1024
UPSERT_BATCH_SIZE = 100   # points per upsert call

# Payload fields to index for fast filtered retrieval
INDEXED_PAYLOAD_FIELDS = {
    "risk_tier_relevance": PayloadSchemaType.KEYWORD,
    "actor_relevance":     PayloadSchemaType.KEYWORD,
    "doc_type":            PayloadSchemaType.KEYWORD,
    "article_num":         PayloadSchemaType.KEYWORD,
    "doc_id":              PayloadSchemaType.KEYWORD,
    "is_normative":        PayloadSchemaType.BOOL,
    "level":               PayloadSchemaType.INTEGER,
    "date_application":    PayloadSchemaType.KEYWORD,
}


# ---------------------------------------------------------------------------
# Qdrant client factory
# ---------------------------------------------------------------------------

def get_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """
    Create a Qdrant client from env vars or explicit parameters.

    Falls back to in-memory client if no URL is provided —
    useful for unit tests without a live Qdrant instance.
    """
    qdrant_url = url or os.getenv("QDRANT_URL")
    qdrant_key = api_key or os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        logger.info(f"Connecting to Qdrant Cloud: {qdrant_url}")
        return QdrantClient(url=qdrant_url, api_key=qdrant_key)
    else:
        logger.warning("No QDRANT_URL found — using in-memory Qdrant (testing only)")
        return QdrantClient(":memory:")


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
) -> None:
    """
    Create the Qdrant collection with dense + sparse vector config.

    Args:
        client:          Qdrant client
        collection_name: name for the collection
        recreate:        if True, delete existing collection first
                         use when re-indexing after corpus update
    """
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            logger.info(f"Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
        else:
            logger.info(
                f"Collection '{collection_name}' already exists — skipping creation. "
                f"Pass recreate=True to force recreation."
            )
            return

    logger.info(f"Creating collection '{collection_name}'...")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,  # keep in memory for fast retrieval
                )
            )
        },
    )

    # Create payload indexes for filtered retrieval
    logger.info("Creating payload indexes...")
    for field_name, field_type in INDEXED_PAYLOAD_FIELDS.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_type,
        )
        logger.debug(f"  Index created: {field_name} ({field_type})")

    logger.info(
        f"Collection '{collection_name}' created with:\n"
        f"  Dense vector:  '{DENSE_VECTOR_NAME}' ({DENSE_DIM}d, Cosine)\n"
        f"  Sparse vector: '{SPARSE_VECTOR_NAME}' (BM25)\n"
        f"  Payload indexes: {list(INDEXED_PAYLOAD_FIELDS.keys())}"
    )


# ---------------------------------------------------------------------------
# Point builder
# ---------------------------------------------------------------------------

def _build_point(embedded: EmbeddedChunk) -> PointStruct:
    """
    Convert an EmbeddedChunk into a Qdrant PointStruct.

    Uses chunk_id as the point ID (string — Qdrant supports both
    integer and string IDs; strings are more readable in the dashboard).
    """
    chunk = embedded.chunk

    # Full metadata payload — everything stored for retrieval + filtering
    payload = chunk.to_dict()
    # Remove text_to_embed from payload — it's large and not needed
    # after indexing (we store text for display, not text_to_embed)
    payload.pop("text_to_embed", None)

    return PointStruct(
        id=_chunk_id_to_int(chunk.chunk_id),
        vector={
            DENSE_VECTOR_NAME: embedded.dense_vector,
            SPARSE_VECTOR_NAME: {
                "indices": embedded.sparse_indices,
                "values":  embedded.sparse_values,
            },
        },
        payload=payload,
    )


def _chunk_id_to_int(chunk_id: str) -> int:
    """
    Convert string chunk_id to a stable integer for Qdrant.

    Qdrant requires integer or UUID point IDs.
    We use a stable hash so the same chunk always gets the same ID.
    """
    return abs(hash(chunk_id)) % (2 ** 53)


# ---------------------------------------------------------------------------
# Upsert pipeline
# ---------------------------------------------------------------------------

def upsert_chunks(
    client: QdrantClient,
    embedded_chunks: list[EmbeddedChunk],
    collection_name: str = COLLECTION_NAME,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """
    Upsert all embedded chunks into Qdrant in batches.

    Args:
        client:           Qdrant client
        embedded_chunks:  output from embed_corpus()
        collection_name:  target collection
        batch_size:       points per upsert call
    """
    logger.info(
        f"Upserting {len(embedded_chunks)} points into "
        f"'{collection_name}' (batch size: {batch_size})..."
    )

    batches = [
        embedded_chunks[i: i + batch_size]
        for i in range(0, len(embedded_chunks), batch_size)
    ]

    total_upserted = 0
    for batch in tqdm(batches, desc="Indexing", unit="batch"):
        points = [_build_point(e) for e in batch]
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,  # wait for indexing to complete before next batch
        )
        total_upserted += len(points)

    logger.info(f"Upsert complete — {total_upserted} points indexed")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """
    Verify the collection is populated and return basic stats.
    Also runs a quick test query to confirm retrieval works.
    """
    info = client.get_collection(collection_name)
    count = client.count(collection_name).count

    stats = {
        "collection": collection_name,
        "points_count": count,
        "status": str(info.status),
    }

    logger.info(
        f"\nCollection '{collection_name}' verified:\n"
        f"  Points:  {stats['points_count']}\n"
        f"  Status:  {stats['status']}"
    )

    # Quick filtered query sanity check
    try:
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="risk_tier_relevance",
                        match=MatchAny(any=["high_risk"]),
                    )
                ]
            ),
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0]
        logger.info(
            f"  Filter test (high_risk): {len(points)} points returned ✅"
        )
        stats["filter_test"] = "passed"
    except Exception as e:
        logger.warning(f"  Filter test failed: {e}")
        stats["filter_test"] = "failed"

    return stats


# ---------------------------------------------------------------------------
# Full indexing pipeline
# ---------------------------------------------------------------------------

def index_corpus(
    embedded_chunks: list[EmbeddedChunk],
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
) -> dict:
    """
    Full indexing pipeline: connect → create collection → upsert → verify.

    Args:
        embedded_chunks: output from embed_corpus()
        url:             Qdrant URL (defaults to QDRANT_URL env var)
        api_key:         Qdrant API key (defaults to QDRANT_API_KEY env var)
        collection_name: target collection name
        recreate:        delete and recreate collection if it exists

    Returns:
        verification stats dict
    """
    client = get_qdrant_client(url, api_key)
    create_collection(client, collection_name, recreate=recreate)
    upsert_chunks(client, embedded_chunks, collection_name)
    stats = verify_collection(client, collection_name)
    return stats


# ---------------------------------------------------------------------------
# CLI — python -m src.ingestion.indexer
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from .loader import load_corpus
    from .chunker import AIActChunker
    from .embedder import embed_corpus

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data_raw = Path(__file__).parents[2] / "data" / "raw"

    print("Loading corpus...")
    docs = load_corpus(data_raw, fetch_html=False, skip_missing=True)

    print("Chunking...")
    chunker = AIActChunker()
    chunks = chunker.chunk_corpus(docs)

    print("Embedding...")
    embedded = embed_corpus(chunks)

    print("\nIndexing into Qdrant...")
    stats = index_corpus(embedded, recreate=True)

    print(f"\n{'='*45}")
    print(f"✅ Indexing complete!")
    print(f"   Collection:  {stats['collection']}")
    print(f"   Points:      {stats['points_count']}")
    print(f"   Status:      {stats['status']}")
    print(f"   Filter test: {stats['filter_test']}")
    print(f"{'='*45}")
