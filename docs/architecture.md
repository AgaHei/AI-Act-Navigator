# Architecture decisions

This document records the key design decisions for AI Act Navigator and the reasoning behind them.

---

## ADR-001: LangChain + LangGraph over LlamaIndex

**Decision:** Use LangChain for retrieval components and LangGraph for the agentic pipeline.

**Reasoning:**
- LangGraph is the most mature framework for conditional multi-step agentic workflows as of 2025–2026
- LangChain has stronger recruiter recognition in the Paris AI market
- Native `langchain-mistralai` integration with no adapter layer needed
- LangGraph's state graph model maps cleanly to our classify → retrieve → map → plan pipeline

**Trade-off:** LlamaIndex has a more opinionated RAG abstraction that would be faster to scaffold. Accepted in favour of more explicit control over the retrieval pipeline.

---

## ADR-002: mistral-embed over OpenAI text-embedding-3

**Decision:** Use `mistral-embed` for dense vector generation.

**Reasoning:**
- Single provider (Mistral throughout) simplifies architecture and billing
- GDPR-compliant EU hosting — directly relevant for an EU regulatory compliance tool
- 1024-dim vectors are sufficient for this corpus size
- Competitive quality for legal English text

**Trade-off:** OpenAI `text-embedding-3-small` scores marginally better on some English benchmarks. Accepted — the difference is not material for this corpus.

---

## ADR-003: Hybrid search (dense + BM25) as mandatory baseline

**Decision:** Never use dense-only retrieval. Hybrid search with BM25 is the minimum viable retrieval strategy.

**Reasoning:**
- Legal text contains exact references ("Article 9(4)(b)") that dense embeddings miss
- BM25 catches verbatim legal language; dense catches semantic intent
- The combination is strictly better than either alone on legal corpora
- RAGAS evaluation will quantify the improvement — this is a core portfolio demonstration

---

## ADR-004: Three-level hierarchical chunking

**Decision:** Chunk at section → article → paragraph levels, with context augmentation headers.

**Reasoning:**
- Legal articles are semantically coherent at the paragraph level
- Cross-references are frequent — context headers ensure embedded vectors capture hierarchical position
- Level-1 (article) chunks answer "what does this article say overall"
- Level-2 (paragraph) chunks answer "what does this specific provision require"
- Recitals are stored separately with `related_articles[]` metadata for optional retrieval

---

## ADR-005: Confidence-gated clarification loop (max 2 rounds)

**Decision:** LLM extractor scores confidence per field; targeted clarification if critical fields below threshold; hard cap at 2 rounds.

**Reasoning:**
- Silent misclassification is worse than asking one follow-up question
- Surgical follow-ups (about the specific uncertain field) are UX-acceptable
- Hard cap prevents the tool feeling like an interrogation
- Uncertain fields are flagged in the final report regardless — transparency about confidence is a feature

---

## ADR-006: Qdrant Cloud free tier for deployment

**Decision:** Use Qdrant Cloud free tier (1GB RAM, 4GB disk) for both development and production.

**Reasoning:**
- The full corpus vectorized (AI Act + GPAI documents) fits comfortably within 1GB
- No credit card required for free tier — reduces friction for portfolio reviewers wanting to test locally
- Qdrant natively supports both dense and sparse vectors in the same collection — no second vector store needed

---

## ADR-007: Local cross-encoder reranker

**Decision:** Use `cross-encoder/ms-marco-MiniLM-L-6-v2` from HuggingFace, downloaded at Docker build time.

**Reasoning:**
- Zero API cost per query — critical for keeping demo costs low
- Downloaded at Docker build time → no cold-start latency in production
- Quality is sufficient for top-10 → top-4 reranking on this corpus
- Cohere Rerank would be higher quality but adds API dependency and cost

---

## ADR-008: Split generation between mistral-small and mistral-large

**Decision:** Use `mistral-small` for extraction step, `mistral-large` for classification + obligation mapping + action planning.

**Reasoning:**
- Extraction (free-text → structured JSON) is a simple structured output task — small model is sufficient
- Classification and legal reasoning require the stronger model
- Cost saving: ~80% reduction on the extraction step with no quality loss
- Makes the cost per assessment ~$0.008 instead of ~$0.012
