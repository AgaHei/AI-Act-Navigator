"""
hybrid.py — Hybrid retrieval for AI Act Navigator

Combines dense (semantic) and sparse (BM25) retrieval using
Reciprocal Rank Fusion (RRF) — the standard score fusion method
for hybrid search.

Why RRF over weighted sum:
  - Dense and sparse scores are on different scales (cosine vs BM25)
  - RRF uses rank positions instead of raw scores, making it scale-invariant
  - Works well without tuning — k=60 is the standard default
  - Consistently outperforms weighted sum on legal/technical corpora

Pipeline:
  query → dense retrieval (top_k * 2)
        → sparse retrieval (top_k * 2)
        → RRF score fusion
        → deduplicate
        → return top_k merged results
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from .dense import (
    DenseRetriever,
    QueryEmbedder,
    RetrievedChunk,
    build_filter,
    get_dense_retriever,
)
from .sparse import SparseRetriever, VocabularyStore, get_sparse_retriever

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10
RRF_K = 60          # standard RRF constant — lower = more weight to top ranks


# ---------------------------------------------------------------------------
# RRF score fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """
    Merge dense and sparse results using Reciprocal Rank Fusion.

    RRF score for a document d:
        RRF(d) = Σ 1 / (k + rank(d, list_i))

    where rank is 1-indexed position in each result list.
    Documents appearing in both lists get higher scores.

    Args:
        dense_results:  ranked list from dense retriever
        sparse_results: ranked list from sparse retriever
        k:              RRF constant (default 60)

    Returns:
        merged list sorted by descending RRF score,
        with retrieval_method set to "hybrid"
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    # Score from dense results
    for rank, chunk in enumerate(dense_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0)
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    # Score from sparse results
    for rank, chunk in enumerate(sparse_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0)
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
        if chunk.chunk_id not in chunk_map:
            chunk_map[chunk.chunk_id] = chunk

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    merged = []
    for cid in sorted_ids:
        chunk = chunk_map[cid]
        # Create new chunk with RRF score and hybrid method tag
        merged.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            score=rrf_scores[cid],
            text=chunk.text,
            context_header=chunk.context_header,
            doc_id=chunk.doc_id,
            doc_type=chunk.doc_type,
            article_num=chunk.article_num,
            article_name=chunk.article_name,
            paragraph_num=chunk.paragraph_num,
            annex_id=chunk.annex_id,
            chapter_name=chunk.chapter_name,
            section_name=chunk.section_name,
            risk_tier_relevance=chunk.risk_tier_relevance,
            actor_relevance=chunk.actor_relevance,
            cross_references=chunk.cross_references,
            date_application=chunk.date_application,
            is_normative=chunk.is_normative,
            level=chunk.level,
            retrieval_method="hybrid",
        ))

    return merged


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines dense and sparse retrieval with RRF score fusion.

    This is the primary retrieval strategy for AI Act Navigator.
    The dense retriever handles semantic queries; the sparse retriever
    handles exact legal references; RRF merges both result lists.

    The cross-reference resolver in the agentic layer runs on top
    of hybrid results to follow Art. → Annex chains.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
    ):
        self._dense = dense_retriever
        self._sparse = sparse_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        risk_tiers: Optional[list[str]] = None,
        actors: Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
        normative_only: bool = False,
        fetch_multiplier: int = 2,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-k chunks using hybrid dense + sparse search.

        Fetches top_k * fetch_multiplier from each retriever before
        fusion, then returns the top_k after RRF scoring.
        The multiplier ensures both retrievers contribute enough
        candidates to the fusion pool.

        Args:
            query:            natural language query
            top_k:            final number of chunks to return
            risk_tiers:       optional payload filter
            actors:           optional payload filter
            doc_types:        optional payload filter
            normative_only:   exclude non-normative chunks
            fetch_multiplier: how many × top_k to fetch before fusion

        Returns:
            list of RetrievedChunk sorted by RRF score, tagged "hybrid"
        """
        fetch_k = top_k * fetch_multiplier

        shared_kwargs = dict(
            risk_tiers=risk_tiers,
            actors=actors,
            doc_types=doc_types,
            normative_only=normative_only,
        )

        # Run both retrievers
        dense_results = self._dense.retrieve(query, top_k=fetch_k, **shared_kwargs)
        sparse_results = self._sparse.retrieve(query, top_k=fetch_k, **shared_kwargs)

        logger.debug(
            f"Pre-fusion: dense={len(dense_results)}, "
            f"sparse={len(sparse_results)} results"
        )

        # Fuse with RRF
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        # Return top_k
        results = fused[:top_k]

        logger.debug(
            f"Post-fusion: {len(results)} results returned "
            f"(from {len(fused)} unique chunks)"
        )

        return results

    def retrieve_with_breakdown(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        **kwargs,
    ) -> dict:
        """
        Retrieve with full breakdown for RAGAS evaluation and debugging.

        Returns both the merged results and the individual dense/sparse
        results for comparison.
        """
        fetch_k = top_k * 2
        dense_results = self._dense.retrieve(query, top_k=fetch_k, **kwargs)
        sparse_results = self._sparse.retrieve(query, top_k=fetch_k, **kwargs)
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        return {
            "query": query,
            "hybrid": fused[:top_k],
            "dense_only": dense_results[:top_k],
            "sparse_only": sparse_results[:top_k],
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "unique_after_fusion": len(fused),
            "overlap": len(
                {r.chunk_id for r in dense_results} &
                {r.chunk_id for r in sparse_results}
            ),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_hybrid_retriever(
    collection_name: str = "ai_act_navigator",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
) -> HybridRetriever:
    """Build a HybridRetriever from environment variables."""
    url = qdrant_url or os.getenv("QDRANT_URL")
    q_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=q_key) if url else QdrantClient(":memory:")

    embedder = QueryEmbedder(api_key=mistral_api_key)
    dense = DenseRetriever(client, embedder, collection_name)
    sparse = SparseRetriever(client, collection_name)

    return HybridRetriever(dense, sparse)


# ---------------------------------------------------------------------------
# CLI — compare dense vs sparse vs hybrid on same queries
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    retriever = get_hybrid_retriever()

    test_queries = [
        "obligations for providers of high-risk AI systems",
        "Article 9 risk management system",
        "transparency requirements conversational AI Article 50",
        "prohibited subliminal manipulation",
        "GPAI model documentation requirements",
    ]

    for query in test_queries:
        breakdown = retriever.retrieve_with_breakdown(query, top_k=3)

        print(f"\n{'='*65}")
        print(f"Query: {query}")
        print(f"Dense/sparse overlap: {breakdown['overlap']} chunks")
        print(f"{'='*65}")

        print("\n  HYBRID (RRF):")
        for i, r in enumerate(breakdown["hybrid"], 1):
            print(f"    [{i}] rrf={r.score:.5f} | {r.display_reference}")

        print("\n  DENSE only:")
        for i, r in enumerate(breakdown["dense_only"], 1):
            print(f"    [{i}] cos={r.score:.4f} | {r.display_reference}")

        print("\n  SPARSE only:")
        for i, r in enumerate(breakdown["sparse_only"], 1):
            print(f"    [{i}] bm25={r.score:.4f} | {r.display_reference}")
