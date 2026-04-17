# AI Act Navigator 🇪🇺

> **Advanced RAG system for EU AI Act compliance assessment**  
> Helping AI consultants classify AI systems by risk tier and generate actionable compliance roadmaps.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-green.svg)](https://langchain.com)
[![Mistral AI](https://img.shields.io/badge/LLM-Mistral_AI-orange.svg)](https://mistral.ai)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech)
[![HuggingFace](https://img.shields.io/badge/Deploy-HuggingFace_Spaces-yellow.svg)](https://huggingface.co/spaces)

---

## What it does

AI Act Navigator takes a description of an AI system and produces a structured compliance assessment:

1. **Risk classification** — maps the system to the correct AI Act risk tier (prohibited / high-risk / limited-risk / minimal-risk) with full legal reasoning and article references
2. **Obligation mapping** — identifies every applicable article, annex, and requirement for the deployer/provider
3. **Action plan** — generates a prioritised checklist of compliance steps with deadlines drawn from the AI Act's phased implementation timeline

**Example input:**
> *"Complice is a RAG-based conversational assistant for young adults aged 16–25, deployed on a public web platform, built on Mistral AI, offering emotional support and information on mental health topics."*

**Example output:** Risk tier → Limited Risk with Art. 50 transparency obligations + precautionary Annex III point 4 assessment recommended. Action plan with 6 prioritised steps, deadlines, and legal anchors.

---

## Architecture

```
Free-text input
      ↓
LLM Extractor (mistral-small)        ← pulls structured attributes
      ↓
Structured intake form
      ↓ [confidence gate — clarification loop if needed]
Classification Engine                ← Art. 5, Art. 6, Annex I, Annex III
      ↓
Hybrid Retrieval Pipeline
  ├── Dense retrieval  (mistral-embed + Qdrant)
  ├── Sparse retrieval (BM25)
  └── Cross-ref resolver (agentic — follows Art. → Annex links)
      ↓
Cross-encoder Reranker
      ↓
Agentic Obligation Mapper            ← conditional retrieval by risk tier
      ↓
Action Plan Generator                ← deadlines from implementation timeline
      ↓
Compliance Assessment Report
```

### Key technical features

| Feature | Implementation |
|---------|---------------|
| **Hierarchical chunking** | 3-level schema (section → article → paragraph) with context augmentation headers |
| **Hybrid search** | Dense (mistral-embed) + Sparse (BM25) combined in Qdrant, reranked by cross-encoder |
| **Cross-reference resolution** | Agentic layer follows `Art. X → Annex Y` chains at retrieval time |
| **Confidence-gated clarification** | LLM scores field confidence; targeted follow-up questions if below threshold |
| **Structured metadata filtering** | Qdrant payload filters on `risk_tier`, `actor`, `date_application` |
| **RAGAS evaluation** | Quantitative benchmarking of context precision, recall, faithfulness, answer relevance |
| **Agentic pipeline** | LangGraph multi-step graph: extract → classify → retrieve → map → plan |

---

## Tech stack

| Layer | Tool |
|-------|------|
| LLM (reasoning) | `mistral-large-3` via Mistral AI API |
| LLM (extraction) | `mistral-small-3.2` via Mistral AI API |
| Embeddings | `mistral-embed` (1024 dims) |
| Sparse encoding | `rank_bm25` |
| Vector store | Qdrant Cloud (free tier) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Orchestration | LangChain + LangGraph |
| Evaluation | RAGAS |
| UI | Streamlit |
| Deployment | HuggingFace Spaces (Dockerfile) |

---

## Corpus

| Document | Source | Status |
|----------|--------|--------|
| AI Act full text (Regulation EU 2024/1689) | EUR-Lex | ✅ Included |
| Annexes I–XIII | EUR-Lex | ✅ Included |
| GPAI Code of Practice (July 2025) | EU AI Office | ✅ Included |
| GPAI Guidelines — Commission (July 2025) | EU AI Office | ✅ Included |

See [`docs/corpus_inventory.md`](docs/corpus_inventory.md) for the full corpus inventory, chunking strategy, and metadata schema.

---

## Quickstart

### Prerequisites

- Python 3.11+
- Mistral AI API key ([console.mistral.ai](https://console.mistral.ai))
- Qdrant Cloud account — free tier, no credit card ([cloud.qdrant.io](https://cloud.qdrant.io))

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-act-navigator.git
cd ai-act-navigator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Ingest the corpus

```bash
python scripts/ingest.py
# Downloads AI Act from EUR-Lex, chunks, embeds, and indexes into Qdrant
# Estimated cost: < $0.10 | Duration: ~5 min
```

### Run the app

```bash
streamlit run src/ui/app.py
```

### Run RAGAS evaluation

```bash
python src/evaluation/ragas_runner.py
# Outputs precision/recall/faithfulness scores per retrieval strategy
```

---

## Project structure

```
ai-act-navigator/
├── README.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── docs/
│   ├── corpus_inventory.md      # Corpus structure, chunking strategy, metadata schema
│   └── architecture.md          # Detailed architecture decisions
├── data/
│   ├── raw/                     # Downloaded source documents (gitignored)
│   ├── processed/               # Chunked documents with metadata
│   └── embeddings/              # Cached embeddings (gitignored)
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # EUR-Lex + EU AI Office document fetcher
│   │   ├── chunker.py           # Hierarchical chunking (3-level schema)
│   │   ├── embedder.py          # mistral-embed + BM25 sparse vectors
│   │   └── indexer.py           # Qdrant collection setup + upsert
│   ├── retrieval/
│   │   ├── dense.py             # Qdrant dense vector search
│   │   ├── sparse.py            # BM25 sparse search
│   │   ├── hybrid.py            # Combined hybrid retriever
│   │   └── reranker.py          # Cross-encoder reranking
│   ├── agents/
│   │   ├── extractor.py         # Free-text → structured form (with confidence scoring)
│   │   ├── classifier.py        # Risk tier classification (Art. 5, 6, Annex III)
│   │   ├── obligation_mapper.py # Conditional retrieval by risk tier
│   │   ├── action_planner.py    # Compliance checklist + deadline generation
│   │   └── graph.py             # LangGraph pipeline definition
│   ├── evaluation/
│   │   ├── ragas_runner.py      # RAGAS metrics computation
│   │   └── test_dataset.py      # 30 curated Q&A pairs for evaluation
│   └── ui/
│       ├── app.py               # Streamlit main app
│       └── components.py        # Reusable UI components
├── notebooks/
│   ├── 01_corpus_exploration.ipynb
│   ├── 02_chunking_strategy.ipynb
│   ├── 03_retrieval_comparison.ipynb
│   └── 04_ragas_evaluation.ipynb
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── test_agents.py
└── scripts/
    └── ingest.py                # One-shot corpus ingestion script
```

---

## RAGAS Evaluation

Retrieval strategies are benchmarked on a curated test dataset of 30 question/ground-truth pairs covering:

- Prohibited system queries (Art. 5)
- Risk classification (Art. 6 + Annex III)
- High-risk obligations (Art. 9, 10, 13, 17)
- Transparency obligations (Art. 50)
- GPAI queries (Art. 53, 55)
- Cross-reference chains

| Metric | Description |
|--------|-------------|
| `context_precision` | Are retrieved chunks relevant to the query? |
| `context_recall` | Are all necessary chunks retrieved? |
| `faithfulness` | Does the answer stay grounded in retrieved content? |
| `answer_relevance` | Does the answer address the question? |

Results comparing dense-only vs hybrid vs hybrid+reranking are documented in `notebooks/04_ragas_evaluation.ipynb`.

---

## AI Act Implementation Timeline

Key deadlines surfaced in action plans:

| Date | What applies |
|------|-------------|
| ✅ 2 Feb 2025 | Prohibited AI practices (Art. 5) — **in force** |
| ✅ 2 Aug 2025 | GPAI model obligations (Art. 51–56) — **in force** |
| ⏳ 2 Aug 2026 | High-risk AI obligations (Annex III systems) — **4 months away** |
| ⏳ 2 Aug 2027 | High-risk AI embedded in regulated products |

---

## Use case: Complice

The worked example used throughout development: *Complice*, a RAG-based conversational assistant for young adults built on Mistral AI.

**Classification result:** Limited Risk — Art. 50 transparency obligations apply. Precautionary Annex III point 4 (education/vulnerable users) assessment recommended and documented.

**Action plan generated:** 5 compliance steps, all with deadline 2 Aug 2026 or earlier.

---

## Cost

Total API cost for the full development cycle (ingestion + testing + RAGAS evaluation + portfolio demos): **< €20**.

| Item | Cost |
|------|------|
| Corpus ingestion (mistral-embed) | < $0.10 |
| Development queries (~100) | ~$0.80 |
| RAGAS evaluation runs | ~$1.20 |
| Portfolio demos | ~$0.40 |
| Qdrant Cloud (free tier) | $0 |
| HuggingFace Spaces (free tier) | $0 |

---

## Author

**Aga Heijligers** — AI Consultant · ML Engineer · Multilingual  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)  

Certified AI Architect (Jedha Bootcamp, Paris) with a background in international B2B, key account management, and professional translation.  
Targeting roles in AI consulting, solutions engineering, and technical account management in the Paris market.
