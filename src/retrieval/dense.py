"""
dense.py — Dense vector retrieval for AI Act Navigator

Queries the Qdrant collection using mistral-embed dense vectors.
Supports optional payload filtering so the classification engine
can pre-filter by risk tier, actor, or date before semantic search.

This is the baseline retrieval strategy — hybrid.py combines it
with sparse (BM25) retrieval for better precision on legal references.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from mistralai import Mistral
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

load_dotenv()
logger = logging.getLogger(__name__)

MISTRAL_EMBED_MODEL = "mistral-embed"
DENSE_VECTOR_NAME = "dense"
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# Data model — shared across dense, sparse, hybrid
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk with its relevance score and metadata.
    Shared data model used by dense, sparse, hybrid, and reranker.
    """
    chunk_id: str
    score: float
    text: str
    context_header: str
    doc_id: str
    doc_type: str
    article_num: Optional[str]
    article_name: Optional[str]
    paragraph_num: Optional[str]
    annex_id: Optional[str]
    chapter_name: Optional[str]
    section_name: Optional[str]
    risk_tier_relevance: list[str]
    actor_relevance: list[str]
    cross_references: list[str]
    date_application: Optional[str]
    is_normative: bool
    level: int
    retrieval_method: str = "dense"   # "dense" | "sparse" | "hybrid"

    def __repr__(self) -> str:
        loc = self.article_num or self.annex_id or self.doc_id
        return (
            f"RetrievedChunk(id={self.chunk_id!r}, "
            f"score={self.score:.4f}, "
            f"loc={loc!r}, "
            f"method={self.retrieval_method!r})"
        )

    @property
    def display_reference(self) -> str:
        """Human-readable legal reference for display in UI."""
        parts = []
        if self.article_num:
            ref = f"Article {self.article_num}"
            if self.paragraph_num:
                ref += f"({self.paragraph_num})"
            if self.article_name:
                ref += f" — {self.article_name}"
            parts.append(ref)
        elif self.annex_id:
            parts.append(f"Annex {self.annex_id.replace('annex_', '').upper()}")
        if self.chapter_name:
            parts.append(self.chapter_name)
        return " | ".join(parts) if parts else self.chunk_id


# ---------------------------------------------------------------------------
# Query embedder — reused by dense and hybrid retrievers
# ---------------------------------------------------------------------------

class QueryEmbedder:
    """Embeds query strings using mistral-embed."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise ValueError("MISTRAL_API_KEY not set")
        self._client = Mistral(api_key=key)

    def embed(self, query: str) -> list[float]:
        """Embed a single query string → 1024-dim vector."""
        response = self._client.embeddings.create(
            model=MISTRAL_EMBED_MODEL,
            inputs=[query],
        )
        return response.data[0].embedding

    def embed_batch(self, queries: list[str]) -> list[list[float]]:
        """Embed multiple queries in one API call."""
        response = self._client.embeddings.create(
            model=MISTRAL_EMBED_MODEL,
            inputs=queries,
        )
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Payload filter builder — shared utility
# ---------------------------------------------------------------------------

def build_filter(
    risk_tiers: Optional[list[str]] = None,
    actors: Optional[list[str]] = None,
    doc_types: Optional[list[str]] = None,
    normative_only: bool = False,
) -> Optional[Filter]:
    """
    Build a Qdrant payload filter from retrieval constraints.

    Args:
        risk_tiers:     e.g. ["high_risk"] — only chunks relevant to this tier
        actors:         e.g. ["provider"] — only chunks addressing this actor
        doc_types:      e.g. ["regulation"] — exclude GPAI docs if needed
        normative_only: if True, exclude recitals and guidelines

    Returns:
        Qdrant Filter object, or None if no constraints
    """
    conditions = []

    if risk_tiers:
        conditions.append(
            FieldCondition(
                key="risk_tier_relevance",
                match=MatchAny(any=risk_tiers + ["all"]),
            )
        )

    if actors:
        conditions.append(
            FieldCondition(
                key="actor_relevance",
                match=MatchAny(any=actors + ["all"]),
            )
        )

    if doc_types:
        conditions.append(
            FieldCondition(
                key="doc_type",
                match=MatchAny(any=doc_types),
            )
        )

    if normative_only:
        conditions.append(
            FieldCondition(
                key="is_normative",
                match=MatchValue(value=True),
            )
        )

    if not conditions:
        return None

    from qdrant_client.models import Filter as QFilter
    return QFilter(must=conditions)


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Retrieves chunks using cosine similarity on mistral-embed vectors.

    This is the semantic search layer — finds chunks that are
    conceptually similar to the query even without exact term overlap.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        query_embedder: QueryEmbedder,
        collection_name: str = "ai_act_navigator",
    ):
        self._client = qdrant_client
        self._embedder = query_embedder
        self._collection = collection_name

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        risk_tiers: Optional[list[str]] = None,
        actors: Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
        normative_only: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-k chunks most semantically similar to the query.

        Args:
            query:         natural language query
            top_k:         number of chunks to return
            risk_tiers:    optional payload filter by risk tier
            actors:        optional payload filter by actor type
            doc_types:     optional payload filter by document type
            normative_only: if True, exclude non-normative chunks

        Returns:
            list of RetrievedChunk sorted by descending score
        """
        # Embed query
        query_vector = self._embedder.embed(query)

        # Build optional filter
        payload_filter = build_filter(
            risk_tiers=risk_tiers,
            actors=actors,
            doc_types=doc_types,
            normative_only=normative_only,
        )

        # Query Qdrant
        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=payload_filter,
            with_payload=True,
        )

        return [
            _point_to_retrieved_chunk(point, method="dense")
            for point in results.points
        ]


# ---------------------------------------------------------------------------
# Point → RetrievedChunk conversion — shared utility
# ---------------------------------------------------------------------------

def _point_to_retrieved_chunk(
    point,
    method: str = "dense",
) -> RetrievedChunk:
    """Convert a Qdrant ScoredPoint to a RetrievedChunk."""
    p = point.payload or {}
    return RetrievedChunk(
        chunk_id=p.get("chunk_id", str(point.id)),
        score=point.score,
        text=p.get("text", ""),
        context_header=p.get("context_header", ""),
        doc_id=p.get("doc_id", ""),
        doc_type=p.get("doc_type", ""),
        article_num=p.get("article_num"),
        article_name=p.get("article_name"),
        paragraph_num=p.get("paragraph_num"),
        annex_id=p.get("annex_id"),
        chapter_name=p.get("chapter_name"),
        section_name=p.get("section_name"),
        risk_tier_relevance=p.get("risk_tier_relevance", []),
        actor_relevance=p.get("actor_relevance", []),
        cross_references=p.get("cross_references", []),
        date_application=p.get("date_application"),
        is_normative=p.get("is_normative", True),
        level=p.get("level", 1),
        retrieval_method=method,
    )


# ---------------------------------------------------------------------------
# Factory — convenience constructor used by hybrid.py and agents
# ---------------------------------------------------------------------------

def get_dense_retriever(
    collection_name: str = "ai_act_navigator",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
) -> DenseRetriever:
    """
    Build a DenseRetriever from environment variables.
    Used by hybrid.py and agent modules.
    """
    from qdrant_client import QdrantClient

    url = qdrant_url or os.getenv("QDRANT_URL")
    q_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=q_key) if url else QdrantClient(":memory:")

    embedder = QueryEmbedder(api_key=mistral_api_key)
    return DenseRetriever(client, embedder, collection_name)


# ---------------------------------------------------------------------------
# CLI — quick retrieval test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    retriever = get_dense_retriever()

    test_queries = [
        "What are the obligations for providers of high-risk AI systems?",
        "transparency requirements for conversational AI",
        "prohibited AI practices subliminal manipulation",
        "Article 9 risk management system",
        "Annex III education high risk",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] score={r.score:.4f} | {r.display_reference}")
            print(f"       {r.text[:150].strip()}...")
