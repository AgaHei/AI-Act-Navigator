"""
reranker.py — Cross-encoder reranking for AI Act Navigator

Takes the top-k results from hybrid retrieval and reranks them
using a cross-encoder model that jointly encodes query + chunk text.

Why reranking matters:
  - Bi-encoders (dense/sparse) encode query and document independently
  - Cross-encoders jointly process (query, document) pairs — much more
    accurate but too slow to run on the full index
  - The standard pipeline: fast bi-encoder retrieval (top-20) →
    slow but accurate cross-encoder reranking (top-4)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Downloaded at Docker build time (no cold-start on HuggingFace Spaces)
  - Fast: ~50ms per (query, doc) pair on CPU
  - Strong: trained on MS MARCO passage ranking, transfers well to legal text
  - Free: local inference, zero API cost
"""

import logging
from typing import Optional

from .dense import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_TOP_K_RERANK = 4        # chunks kept after reranking
DEFAULT_RERANK_CANDIDATES = 10  # chunks passed to reranker from hybrid
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.

    The cross-encoder jointly encodes (query, chunk_text) pairs and
    produces a relevance score that's more accurate than cosine similarity
    or BM25 because it can model fine-grained query-document interactions.

    Model is loaded lazily on first use and cached — loading takes ~2s
    but subsequent reranking calls are fast (~50ms for 10 candidates).
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self._model_name = model_name
        self._model = None   # lazy load

    def _load_model(self):
        """Load cross-encoder model (lazy — only on first rerank call)."""
        if self._model is None:
            logger.info(f"Loading cross-encoder: {self._model_name}")
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                logger.info("Cross-encoder loaded ✅")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = DEFAULT_TOP_K_RERANK,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks by cross-encoder relevance score.

        Args:
            query:  original query string
            chunks: candidate chunks from hybrid retrieval
            top_k:  number of chunks to return after reranking

        Returns:
            top_k chunks sorted by cross-encoder score descending,
            with scores updated to cross-encoder values and
            retrieval_method set to "reranked"
        """
        if not chunks:
            return []

        self._load_model()

        # Build (query, text) pairs for cross-encoder
        # Use context_header + text for richer signal
        pairs = [
            (query, f"{c.context_header}\n\n{c.text}")
            for c in chunks
        ]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach scores and sort
        scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        # Return top_k with updated scores
        results = []
        for score, chunk in scored[:top_k]:
            results.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                score=float(score),
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
                retrieval_method="reranked",
            ))

        logger.debug(
            f"Reranked {len(chunks)} → {len(results)} chunks. "
            f"Top score: {results[0].score:.4f}"
        )

        return results


# ---------------------------------------------------------------------------
# Full retrieval pipeline: hybrid + rerank
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """
    Complete retrieval pipeline: hybrid search → cross-encoder reranking.

    This is the primary interface used by the agentic layer.
    Wraps HybridRetriever + CrossEncoderReranker in one clean call.

    Typical usage:
        pipeline = RetrievalPipeline.from_env()
        results = pipeline.retrieve(
            query="risk management obligations for high-risk AI",
            top_k=4,
            risk_tiers=["high_risk"],
        )
    """

    def __init__(
        self,
        hybrid_retriever,          # HybridRetriever
        reranker: CrossEncoderReranker,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    ):
        self._hybrid = hybrid_retriever
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_RERANK,
        risk_tiers: Optional[list[str]] = None,
        actors: Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
        normative_only: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Full pipeline: hybrid retrieval → cross-encoder reranking.

        Args:
            query:         natural language query
            top_k:         final chunks returned (after reranking)
            risk_tiers:    Qdrant payload filter
            actors:        Qdrant payload filter
            doc_types:     Qdrant payload filter
            normative_only: exclude non-normative content

        Returns:
            top_k RetrievedChunks, reranked by cross-encoder score
        """
        # Step 1 — Hybrid retrieval (fetch more candidates than needed)
        candidates = self._hybrid.retrieve(
            query=query,
            top_k=self._rerank_candidates,
            risk_tiers=risk_tiers,
            actors=actors,
            doc_types=doc_types,
            normative_only=normative_only,
        )

        if not candidates:
            logger.warning(f"No candidates retrieved for query: {query!r}")
            return []

        # Step 2 — Cross-encoder reranking
        reranked = self._reranker.rerank(query, candidates, top_k=top_k)

        logger.info(
            f"Pipeline: hybrid({len(candidates)}) → "
            f"reranked({len(reranked)}) | "
            f"top_score={reranked[0].score:.4f}"
        )

        return reranked

    def retrieve_for_evaluation(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_RERANK,
        **kwargs,
    ) -> dict:
        """
        Retrieve with full breakdown for RAGAS evaluation.
        Returns results from each stage for metric comparison.
        """
        candidates = self._hybrid.retrieve(
            query=query,
            top_k=self._rerank_candidates,
            **kwargs,
        )
        breakdown = self._hybrid.retrieve_with_breakdown(
            query=query,
            top_k=self._rerank_candidates,
            **kwargs,
        )
        reranked = self._reranker.rerank(query, candidates, top_k=top_k)

        return {
            "query": query,
            "reranked": reranked,
            "hybrid": breakdown["hybrid"][:top_k],
            "dense_only": breakdown["dense_only"][:top_k],
            "sparse_only": breakdown["sparse_only"][:top_k],
            "overlap": breakdown["overlap"],
        }

    @classmethod
    def from_env(
        cls,
        collection_name: str = "ai_act_navigator",
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    ) -> "RetrievalPipeline":
        """Build a RetrievalPipeline from environment variables."""
        from .hybrid import get_hybrid_retriever
        hybrid = get_hybrid_retriever(collection_name=collection_name)
        reranker = CrossEncoderReranker()
        return cls(hybrid, reranker, rerank_candidates)


# ---------------------------------------------------------------------------
# CLI — full pipeline test with score comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("Loading pipeline (cross-encoder loads on first query)...")
    pipeline = RetrievalPipeline.from_env()

    test_queries = [
        "What are the risk management obligations for high-risk AI systems?",
        "transparency requirements for AI chatbots interacting with users",
        "Article 9 paragraph 4 risk management measures",
        "prohibited AI practices manipulation vulnerable groups",
    ]

    for query in test_queries:
        print(f"\n{'='*65}")
        print(f"Query: {query}")
        print(f"{'='*65}")

        breakdown = pipeline.retrieve_for_evaluation(query, top_k=4)

        print("\n  RERANKED (final output):")
        for i, r in enumerate(breakdown["reranked"], 1):
            print(f"    [{i}] score={r.score:.4f} | {r.display_reference}")
            print(f"         {r.text[:120].strip()}...")

        print("\n  vs HYBRID (before reranking):")
        for i, r in enumerate(breakdown["hybrid"], 1):
            print(f"    [{i}] rrf={r.score:.5f} | {r.display_reference}")
