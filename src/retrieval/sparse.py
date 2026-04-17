"""
sparse.py — Sparse (BM25) retrieval for AI Act Navigator

Queries the Qdrant collection using BM25 sparse vectors.
Excels at exact legal references ("Article 9(4)(b)"), defined terms,
and acronyms that dense embeddings may dilute.

Used in combination with dense.py in hybrid.py.
"""

import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, NamedSparseVector

from .dense import RetrievedChunk, build_filter, _point_to_retrieved_chunk

load_dotenv()
logger = logging.getLogger(__name__)

SPARSE_VECTOR_NAME = "sparse"
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# Query tokenizer — must match BM25Encoder._tokenize() in embedder.py
# ---------------------------------------------------------------------------

def tokenize_query(text: str) -> list[str]:
    """
    Tokenize a query string for BM25 sparse vector construction.
    Must match the tokenizer used in BM25Encoder._tokenize() exactly,
    otherwise query tokens won't map to the correct vocabulary indices.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9\-]*\b", text.lower())
    return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# Vocabulary loader
# ---------------------------------------------------------------------------

class VocabularyStore:
    """
    Loads and caches the BM25 vocabulary built during embedding.

    The vocabulary maps token → index and must be the same one
    used when building sparse vectors during indexing.
    We rebuild it from the corpus chunks on first use.
    """

    _instance: Optional["VocabularyStore"] = None
    _vocabulary: Optional[dict[str, int]] = None

    @classmethod
    def get(cls, chunks_path: Optional[str] = None) -> "VocabularyStore":
        if cls._instance is None:
            cls._instance = cls(chunks_path)
        return cls._instance

    def __init__(self, chunks_path: Optional[str] = None):
        self._vocabulary = self._load_vocabulary(chunks_path)

    def _load_vocabulary(
        self, chunks_path: Optional[str]
    ) -> dict[str, int]:
        """
        Rebuild vocabulary from saved chunk texts.
        Loads from data/processed/chunks_dump.json by default.
        """
        import json
        from pathlib import Path

        path = chunks_path or str(
            Path(__file__).parents[2] / "data" / "processed" / "chunks_dump.json"
        )

        if not Path(path).exists():
            logger.warning(
                f"chunks_dump.json not found at {path}. "
                f"Sparse retrieval will return empty results."
            )
            return {}

        logger.info(f"Building BM25 vocabulary from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        all_tokens: set[str] = set()
        for chunk in chunks:
            tokens = tokenize_query(chunk.get("text", ""))
            all_tokens.update(tokens)

        vocabulary = {
            token: idx for idx, token in enumerate(sorted(all_tokens))
        }
        logger.info(f"Vocabulary built: {len(vocabulary):,} tokens")
        return vocabulary

    def get_indices(self, tokens: list[str]) -> list[int]:
        """Map tokens to their vocabulary indices."""
        return [
            self._vocabulary[t]
            for t in tokens
            if t in self._vocabulary
        ]

    @property
    def size(self) -> int:
        return len(self._vocabulary) if self._vocabulary else 0


# ---------------------------------------------------------------------------
# Sparse query encoder
# ---------------------------------------------------------------------------

def encode_query_sparse(
    query: str,
    vocabulary: dict[str, int],
) -> tuple[list[int], list[float]]:
    """
    Encode a query as a sparse vector for BM25 retrieval.

    Uses binary term presence (1.0 per unique token) rather than
    TF weighting — appropriate for short queries where term frequency
    is less meaningful than term presence.

    Args:
        query:      raw query string
        vocabulary: token → index mapping from VocabularyStore

    Returns:
        (indices, values) sparse vector representation
    """
    tokens = tokenize_query(query)
    if not tokens:
        return [], []

    # Deduplicate — use binary presence for query terms
    unique_tokens = list(set(tokens))

    indices = []
    values = []
    for token in unique_tokens:
        if token in vocabulary:
            indices.append(vocabulary[token])
            values.append(1.0)

    return indices, values


# ---------------------------------------------------------------------------
# Sparse retriever
# ---------------------------------------------------------------------------

class SparseRetriever:
    """
    Retrieves chunks using BM25 sparse vector similarity.

    Complements dense retrieval by excelling at:
    - Exact article references: "Article 9(4)(b)"
    - Legal defined terms: "high-risk AI system", "GPAI model"
    - Acronyms: "FRIA", "GPAI", "GPAI CoP"
    - Specific obligations: "transparency obligation", "conformity assessment"
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = "ai_act_navigator",
        vocabulary: Optional[dict[str, int]] = None,
    ):
        self._client = qdrant_client
        self._collection = collection_name
        self._vocab = vocabulary or VocabularyStore.get()._vocabulary or {}

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
        Retrieve top-k chunks by BM25 sparse vector similarity.

        Args:
            query:         natural language or exact reference query
            top_k:         number of chunks to return
            risk_tiers:    optional payload filter
            actors:        optional payload filter
            doc_types:     optional payload filter
            normative_only: exclude non-normative chunks if True

        Returns:
            list of RetrievedChunk sorted by descending score
        """
        if not self._vocab:
            logger.warning(
                "BM25 vocabulary is empty — sparse retrieval unavailable. "
                "Run the chunker to generate chunks_dump.json."
            )
            return []

        # Encode query as sparse vector
        indices, values = encode_query_sparse(query, self._vocab)

        if not indices:
            logger.debug(f"No vocabulary matches for query: {query!r}")
            return []

        # Build optional payload filter
        payload_filter = build_filter(
            risk_tiers=risk_tiers,
            actors=actors,
            doc_types=doc_types,
            normative_only=normative_only,
        )

        # Query Qdrant with sparse vector
        results = self._client.query_points(
            collection_name=self._collection,
            query=SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=payload_filter,
            with_payload=True,
        )

        return [
            _point_to_retrieved_chunk(point, method="sparse")
            for point in results.points
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_sparse_retriever(
    collection_name: str = "ai_act_navigator",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> SparseRetriever:
    """Build a SparseRetriever from environment variables."""
    url = qdrant_url or os.getenv("QDRANT_URL")
    q_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=q_key) if url else QdrantClient(":memory:")
    return SparseRetriever(client, collection_name)


# ---------------------------------------------------------------------------
# CLI — quick retrieval test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    retriever = get_sparse_retriever()

    # BM25 shines on exact references — test those specifically
    test_queries = [
        "Article 9 paragraph 4",
        "subliminal manipulation prohibited",
        "GPAI model transparency obligations Article 53",
        "Annex III education vocational training",
        "conformity assessment high-risk",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] score={r.score:.4f} | {r.display_reference}")
            print(f"       {r.text[:150].strip()}...")
