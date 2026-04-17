# AI Act Navigator — Corpus Inventory & Chunking Strategy

> Foundation document for all retrieval architecture decisions.  
> Last updated: April 2026

---

## 1. Corpus Overview

The corpus consists of four source families, each with distinct structure, register, and update cadence. They must be ingested differently and carry different metadata weights.

| # | Source family | Authority | Format | Update frequency |
|---|---------------|-----------|--------|-----------------|
| 1 | AI Act — main text | EUR-Lex (OJ) | HTML / PDF | Static (published 12 Jul 2024) |
| 2 | AI Act — Annexes I–XIII | EUR-Lex (OJ) | HTML / PDF | Static |
| 3 | GPAI Code of Practice | EU AI Office | PDF | Final version: 10 Jul 2025 |
| 4 | GPAI Guidelines (Commission) | EU AI Office | PDF | Published 18 Jul 2025 |

---

## 2. Source URLs

### Primary source — EUR-Lex

| Document | URL | Notes |
|----------|-----|-------|
| AI Act full text (HTML, navigable) | https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689 | Preferred for parsing — clean heading structure |
| AI Act full text (PDF) | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689 | ~144 pages |
| AI Act Official Journal entry | https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng | Canonical reference |

### AI Act Explorer (third-party, useful for navigation)

| Document | URL |
|----------|-----|
| Full text browser | https://artificialintelligenceact.eu/the-act/ |
| Implementation timeline | https://artificialintelligenceact.eu/implementation-timeline/ |

### EU AI Office — operational documents

| Document | URL |
|----------|-----|
| GPAI Code of Practice | https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai |
| GPAI Guidelines (Commission) | https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers |
| AI Office main page | https://digital-strategy.ec.europa.eu/en/policies/ai-office |

---

## 3. AI Act Document Structure

### 3.1 Macro structure

```
REGULATION (EU) 2024/1689
│
├── RECITALS (Whereas clauses — 180 recitals)
│   └── Purpose: legislative intent, interpretation context
│       Register: explanatory, non-normative
│
├── TITLES I–XIII (normative body — 113 articles)
│   ├── Title I     General provisions (Art. 1–4)
│   ├── Title II    Prohibited AI practices (Art. 5)
│   ├── Title III   High-risk AI systems (Art. 6–49)
│   │   ├── Chapter 1  Classification (Art. 6–7)
│   │   ├── Chapter 2  Requirements (Art. 8–15)
│   │   ├── Chapter 3  Obligations providers/deployers (Art. 16–27)
│   │   ├── Chapter 4  Notified bodies (Art. 28–39)
│   │   └── Chapter 5  Registration (Art. 49)
│   ├── Title IV    Transparency obligations (Art. 50)
│   ├── Title V     GPAI models (Art. 51–56)
│   ├── Title VI    Governance (Art. 57–89)
│   ├── Title VII   EU database (Art. 71)
│   ├── Title VIII  Post-market monitoring (Art. 72–74)
│   ├── Title IX    Codes of conduct (Art. 95)
│   ├── Title X     Confidentiality/penalties (Art. 99–101)
│   └── Title XIII  Final provisions (Art. 111–113)
│
└── ANNEXES I–XIII
    ├── Annex I    AI techniques and approaches (Art. 3 ref.)
    ├── Annex II   Union harmonisation legislation — Section A & B
    ├── Annex III  High-risk AI systems (Art. 6(2) ref.) ← KEY
    ├── Annex IV   Technical documentation (Art. 11 ref.)
    ├── Annex V    EU declaration of conformity
    ├── Annex VI   Conformity assessment — internal control
    ├── Annex VII  Conformity assessment — quality management
    ├── Annex VIII Registration information (Art. 49 ref.)
    ├── Annex IX   High-risk AI — Union harmonisation products
    ├── Annex X    Large-scale IT systems
    ├── Annex XI   GPAI transparency info (Art. 53 ref.)
    ├── Annex XII  GPAI summaries (Art. 53 ref.)
    └── Annex XIII Compliance costs (SMEs)
```

### 3.2 Key articles for compliance assessment workflow

| Pipeline step | Primary articles | Key annexes |
|--------------|-----------------|-------------|
| Prohibited check | Art. 5 | — |
| Risk classification | Art. 6, Art. 7 | Annex I, Annex III |
| High-risk obligations | Art. 8–15 (requirements), Art. 16–27 (obligations) | Annex IV |
| Transparency obligations | Art. 50 | — |
| GPAI obligations | Art. 51–56 | Annex XI, Annex XII |
| Registration | Art. 49 | Annex VIII |

---

## 4. Implementation Timeline (metadata: `date_application`)

| Date | What becomes applicable |
|------|------------------------|
| 1 Aug 2024 | AI Act enters into force |
| 2 Feb 2025 | Prohibited AI practices (Art. 5) — **already in force** |
| 2 Feb 2025 | AI literacy obligations — **already in force** |
| 2 Aug 2025 | GPAI model obligations (Art. 51–56) — **already in force** |
| 2 Aug 2026 | High-risk AI obligations (Annex III systems) — **9 months away** |
| 2 Aug 2027 | High-risk AI embedded in regulated products (Annex II Section B) |
| 31 Dec 2030 | Large-scale IT systems (Annex X) |

> This timeline is critical metadata for the action plan generator — every compliance action must carry a `deadline` field derived from this table.

---

## 5. Chunking Strategy

### 5.1 Core principle: respect legal grammar

Legal text has natural semantic units that must not be split across chunks:

- **Article** = smallest self-contained normative unit
- **Paragraph** (numbered 1, 2, 3...) = the standard chunking unit for most articles
- **Point** (lettered a, b, c...) = sub-unit, sometimes bundled with parent paragraph
- **Recital** = standalone interpretive unit
- **Annex entry** = list item (varies: a line, a paragraph, or a full subsection)

### 5.2 Hierarchical chunking schema (three levels)

```
LEVEL 0 — Document section (for context injection only, not retrieved directly)
  Example: "Title III, Chapter 2 — Obligations for High-Risk AI Systems"

LEVEL 1 — Article / Annex section (primary retrieval unit)
  Example: "Article 9 — Risk management system"
  Size: typically 200–600 tokens
  Used when: user asks about an entire obligation

LEVEL 2 — Paragraph / point (fine-grained retrieval unit)
  Example: "Article 9(4)(b)"
  Size: typically 50–150 tokens
  Used when: precise legal reference needed
```

**Rule:** Every Level 2 chunk stores its Level 1 parent as metadata (`parent_article`). Every Level 1 chunk stores its Level 0 context as metadata (`document_section`). This enables parent-augmented retrieval without redundant embeddings.

### 5.3 Annex-specific chunking rules

Annexes are not uniform — each requires a tailored approach:

| Annex | Structure type | Chunking approach |
|-------|---------------|------------------|
| Annex I | Short flat list | One chunk per item + one summary chunk for the full list |
| Annex III | Hierarchical numbered list (8 domains × sub-items) | One chunk per domain (e.g., "Annex III, point 4 — Education") + one chunk per sub-item |
| Annex IV | Form-like documentation requirements | One chunk per section heading |
| Annex VIII | Registration fields | One chunk per category of fields |
| Annex XI–XII | GPAI documentation templates | One chunk per section |

### 5.4 Recital chunking rules

Recitals are stored separately from articles but linked via metadata:

- Each recital = one chunk
- Metadata field `related_articles[]` links recital to the article(s) it contextualises
- At retrieval time, relevant recitals can be optionally appended to article chunks to provide interpretive context (controlled by a retrieval parameter `include_recitals: bool`)

### 5.5 Context augmentation (contextual chunking)

Every chunk receives an auto-generated context header before embedding:

```
[Title III, Chapter 2 — Obligations for High-Risk AI Systems]
[Article 9 — Risk management system | Paragraph 4]

Article 9(4): The risk management measures referred to in paragraph 1 shall be such that any residual risk associated with each hazard...
```

This header is prepended at embedding time only — it is stripped from the displayed output to avoid redundancy in the LLM response.

---

## 6. Metadata Schema

Every chunk in the vector store carries the following metadata payload:

```json
{
  "chunk_id": "art_9_p4_b",
  "doc_type": "article",
  "source": "AI_Act_2024_1689",
  "title_num": "III",
  "title_name": "High-risk AI systems",
  "chapter_num": "2",
  "chapter_name": "Requirements for high-risk AI systems",
  "article_id": "art_9",
  "article_name": "Risk management system",
  "paragraph": "4",
  "point": "b",
  "parent_article": "art_9",
  "document_section": "Title III, Chapter 2",
  "cross_references": ["annex_iv", "art_10", "art_13"],
  "risk_tier_relevance": ["high_risk"],
  "actor_relevance": ["provider", "deployer"],
  "date_application": "2026-08-02",
  "is_normative": true,
  "annex_id": null,
  "text": "..."
}
```

### Additional fields for Annex chunks

```json
{
  "doc_type": "annex",
  "annex_id": "annex_iii",
  "annex_name": "High-risk AI systems",
  "annex_domain": "4",
  "annex_domain_name": "Education and vocational training",
  "annex_sub_item": "a",
  "risk_tier_relevance": ["high_risk"],
  "cross_references": ["art_6", "art_7"]
}
```

### Metadata fields enabling filtered retrieval

| Field | Filter use case |
|-------|----------------|
| `risk_tier_relevance` | "Show only high-risk obligations" |
| `actor_relevance` | "Show only obligations for deployers" |
| `date_application` | "Show obligations already in force" |
| `article_id` | Cross-reference resolution |
| `is_normative` | Exclude recitals from strict legal queries |

---

## 7. Retrieval Strategy Summary

Three complementary strategies are combined at query time:

| Strategy | Tool | Strength |
|----------|------|----------|
| Dense retrieval | Mistral embeddings + Qdrant | Semantic similarity, concept matching |
| Sparse retrieval | BM25 (via Qdrant sparse vectors) | Exact legal references ("Art. 9(4)(b)") |
| Cross-ref resolution | Agentic layer — metadata graph | Following "pursuant to Article X" chains |

Output of all three is passed to a cross-encoder reranker before being sent to the generation step.

---

## 8. RAGAS Evaluation Plan

The following metrics will be tracked per retrieval strategy to enable quantitative comparison:

| Metric | What it measures |
|--------|-----------------|
| `context_precision` | Are retrieved chunks relevant to the query? |
| `context_recall` | Are all necessary chunks retrieved? |
| `faithfulness` | Does the answer stick to retrieved content? |
| `answer_relevance` | Does the answer address the question? |

Test dataset: 20–30 curated question/ground-truth pairs covering:
- Prohibited system queries (Art. 5)
- Risk classification queries (Art. 6 + Annex III)
- Obligation queries (Art. 9, 10, 13, 17)
- Transparency queries (Art. 50)
- GPAI queries (Art. 53, 55)
- Cross-reference chains (Art. → Annex → Art.)

---

## 9. Tech Stack & Cost Estimates

### 9.1 Full technology stack

| Layer | Tool | Model / version | Notes |
|-------|------|----------------|-------|
| Generation LLM | Mistral AI | `mistral-large-3-2512` | Main reasoning — classification + report generation |
| Extraction LLM | Mistral AI | `mistral-small-3.2` | Free-text → structured form parsing (cheaper, fast) |
| Embeddings | Mistral AI | `mistral-embed` | Dense vectors, 1024 dims, 8K context window |
| Sparse encoding | `rank_bm25` (Python) | — | BM25 sparse vectors for hybrid search, runs locally, free |
| Vector store | Qdrant Cloud | Free tier → Standard | Supports dense + sparse natively, no credit card for free tier |
| Reranker | HuggingFace | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder, zero API cost |
| Orchestration | LangChain + LangGraph | `langchain-mistralai` | Agentic multi-step pipeline (see §9.3) |
| Evaluation | RAGAS | `ragas` | Quantitative retrieval metrics |
| UI | Streamlit | — | As planned |
| Deployment | HuggingFace Spaces | Dockerfile | As planned |

### 9.2 Cost breakdown

#### One-time cost: corpus ingestion

The AI Act corpus is small — approximately 150–200 pages total across main text + key annexes.
Estimated total tokens for ingestion: ~300,000 tokens.

| Operation | Model | Tokens | Unit price | Est. cost |
|-----------|-------|--------|-----------|-----------|
| Embed all chunks | `mistral-embed` | ~300K | $0.10 / 1M | ~$0.03 |
| Context header generation | `mistral-small-3.2` | ~300K total | $0.07 in / $0.20 out | ~$0.04 |
| **Total ingestion** | | | | **< $0.10** |

Ingestion is essentially free. You can re-ingest the entire corpus 10× for under $1.

#### Recurring cost: per query (during development)

Each compliance assessment triggers a multi-step agentic pipeline:

| Step | Model | Est. tokens | Est. cost |
|------|-------|------------|-----------|
| Free-text extraction | `mistral-small-3.2` | ~700 total | ~$0.00008 |
| Classification reasoning | `mistral-large-3` | ~2.5K total | ~$0.0018 |
| Obligation retrieval + mapping | `mistral-large-3` | ~3.8K total | ~$0.0027 |
| Action plan generation | `mistral-large-3` | ~5K total | ~$0.0035 |
| **Total per full assessment** | | | **~$0.008 (~0.8 cents)** |

#### Budget scenarios

| Scenario | Queries | Est. cost |
|----------|---------|-----------|
| Development & testing | 100 | ~$0.80 |
| RAGAS evaluation (30 test cases × 5 runs) | 150 | ~$1.20 |
| Portfolio demos (recruiters, interviews) | 50 | ~$0.40 |
| **Total project budget** | | **$5–15** |

A €20 Mistral API credit covers the entire development cycle comfortably.

#### Infrastructure costs

| Service | Plan | Monthly cost |
|---------|------|-------------|
| Qdrant Cloud | Free tier (1GB RAM, 4GB disk) | €0 — sufficient for this corpus |
| HuggingFace Spaces | Free tier (CPU) | €0 |
| Mistral API | Pay-as-you-go | ~€10–15 total for full project |

**Total project cost estimate: €15–20 maximum.**

### 9.3 LangChain vs LlamaIndex

Both are viable. Summary of key differences:

| Criterion | LangChain | LlamaIndex |
|-----------|-----------|------------|
| Mistral integration | Native `langchain-mistralai` | Native `llama-index-llms-mistral` |
| Qdrant + hybrid search | Via custom retriever | Built-in `QdrantVectorStore` with hybrid |
| Agentic workflows | LangGraph (mature, well-documented) | `AgentWorkflow` (newer) |
| RAGAS integration | Seamless | Seamless |
| Portfolio name recognition | Higher | Growing |

**Decision: LangChain + LangGraph** — stronger recruiter recognition and LangGraph is the most mature framework for the multi-step agentic pipeline this project requires.

---

## 10. Out-of-scope for V1

The following are explicitly excluded from the initial corpus to keep scope manageable:

- National implementation acts (each EU member state)
- Harmonised standards (CEN/CENELEC — not yet published as of April 2026)
- Earlier legislative drafts and trilogue documents
- Non-English language versions

These can be added as corpus extensions in V2.
