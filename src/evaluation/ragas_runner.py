"""
ragas_runner.py — RAGAS evaluation for AI Act Navigator

Evaluates retrieval quality across three strategies:
  1. Dense only   (mistral-embed cosine similarity)
  2. Hybrid       (dense + BM25 with RRF fusion)
  3. Reranked     (hybrid + cross-encoder reranking)

Four RAGAS metrics per strategy:
  - context_precision:  are retrieved chunks relevant?
  - context_recall:     are all necessary chunks retrieved?
  - faithfulness:       does the answer stay grounded in context?
  - answer_relevance:   does the answer address the question?

Results are saved to data/processed/ragas_results.json
for analysis in notebook 04.

Estimated cost: ~$1.50-2.00 (28 questions × 3 strategies × LLM scoring)
Estimated time: ~20-30 minutes
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)

# RAGAS imports — graceful failure if not installed
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAGAS not fully installed: {e}")
    RAGAS_AVAILABLE = False

from .test_dataset import get_dataset, EvalSample


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MISTRAL_LARGE_MODEL = os.getenv("MISTRAL_LARGE_MODEL", "mistral-large-latest")
MISTRAL_EMBED_MODEL = "mistral-embed"
RESULTS_PATH = Path(__file__).parents[3] / "data" / "processed" / "ragas_results.json"
TOP_K_RETRIEVE = 4    # chunks passed to each strategy
TOP_K_RERANK = 4      # chunks after reranking


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    """RAGAS scores for a single retrieval strategy."""
    strategy: str
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    per_sample: list[dict] = field(default_factory=list)

    @property
    def average(self) -> float:
        return (
            self.context_precision +
            self.context_recall +
            self.faithfulness +
            self.answer_relevancy
        ) / 4

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "average": round(self.average, 4),
            "per_sample": self.per_sample,
        }


@dataclass
class RAGASReport:
    """Full RAGAS evaluation report across all strategies."""
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    dataset_size: int = 0
    evaluation_date: str = ""
    model_used: str = MISTRAL_LARGE_MODEL
    notes: str = ""

    def winner(self) -> str:
        """Strategy with highest average score."""
        return max(self.strategies, key=lambda k: self.strategies[k].average)

    def improvement(self, baseline: str, improved: str) -> dict:
        """Calculate improvement from baseline to improved strategy."""
        b = self.strategies.get(baseline)
        i = self.strategies.get(improved)
        if not b or not i:
            return {}
        return {
            "context_precision": round(i.context_precision - b.context_precision, 4),
            "context_recall": round(i.context_recall - b.context_recall, 4),
            "faithfulness": round(i.faithfulness - b.faithfulness, 4),
            "answer_relevancy": round(i.answer_relevancy - b.answer_relevancy, 4),
            "average": round(i.average - b.average, 4),
        }

    def to_dict(self) -> dict:
        return {
            "evaluation_date": self.evaluation_date,
            "dataset_size": self.dataset_size,
            "model_used": self.model_used,
            "notes": self.notes,
            "strategies": {k: v.to_dict() for k, v in self.strategies.items()},
            "winner": self.winner(),
            "improvements": {
                "hybrid_vs_dense": self.improvement("dense", "hybrid"),
                "reranked_vs_hybrid": self.improvement("hybrid", "reranked"),
                "reranked_vs_dense": self.improvement("dense", "reranked"),
            },
        }


# ---------------------------------------------------------------------------
# Answer generator — uses the full pipeline to generate answers
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    contexts: list[str],
    api_key: Optional[str] = None,
) -> str:
    """
    Generate an answer from retrieved contexts using mistral-large.
    Used to produce the 'answer' field required by RAGAS.
    """
    from mistralai import Mistral
    key = api_key or os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=key)

    context_text = "\n\n".join([
        f"[Context {i+1}]\n{ctx}"
        for i, ctx in enumerate(contexts[:4])
    ])

    response = client.chat.complete(
        model=MISTRAL_LARGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an EU AI Act compliance expert. "
                    "Answer the question based ONLY on the provided context. "
                    "Be precise and cite specific articles where relevant. "
                    "If the context does not contain enough information, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ],
        temperature=0.1,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Retrieval runners — one per strategy
# ---------------------------------------------------------------------------

def retrieve_dense(question: str, retriever) -> tuple[list[str], list[str]]:
    """Retrieve using dense-only strategy."""
    results = retriever._hybrid._dense.retrieve(question, top_k=TOP_K_RETRIEVE)
    contexts = [r.text for r in results]
    chunk_ids = [r.chunk_id for r in results]
    return contexts, chunk_ids


def retrieve_hybrid(question: str, retriever) -> tuple[list[str], list[str]]:
    """Retrieve using hybrid (dense + sparse + RRF) strategy."""
    results = retriever._hybrid.retrieve(question, top_k=TOP_K_RETRIEVE)
    contexts = [r.text for r in results]
    chunk_ids = [r.chunk_id for r in results]
    return contexts, chunk_ids


def retrieve_reranked(question: str, retriever) -> tuple[list[str], list[str]]:
    """Retrieve using full pipeline with reranking."""
    results = retriever.retrieve(question, top_k=TOP_K_RERANK)
    contexts = [r.text for r in results]
    chunk_ids = [r.chunk_id for r in results]
    return contexts, chunk_ids


# ---------------------------------------------------------------------------
# RAGAS evaluation runner
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    strategy_name: str,
    retrieval_fn,
    pipeline,
    dataset: list[EvalSample],
    llm,
    embeddings,
) -> StrategyResult:
    """
    Run RAGAS evaluation for a single retrieval strategy.

    Args:
        strategy_name:  "dense" | "hybrid" | "reranked"
        retrieval_fn:   function(question, pipeline) → (contexts, chunk_ids)
        pipeline:       RetrievalPipeline instance
        dataset:        list of EvalSample
        llm:            LangchainLLMWrapper for RAGAS scoring
        embeddings:     LangchainEmbeddingsWrapper for RAGAS scoring

    Returns:
        StrategyResult with all four metrics
    """
    logger.info(f"Evaluating strategy: {strategy_name} ({len(dataset)} samples)")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    per_sample = []

    for sample in tqdm(dataset, desc=f"{strategy_name}"):
        try:
            # Retrieve contexts
            contexts, chunk_ids = retrieval_fn(sample.question, pipeline)

            # Generate answer from contexts
            answer = generate_answer(sample.question, contexts)

            questions.append(sample.question)
            answers.append(answer)
            contexts_list.append(contexts)
            ground_truths.append(sample.ground_truth)

            per_sample.append({
                "sample_id": sample.sample_id,
                "category": sample.category,
                "difficulty": sample.difficulty,
                "question": sample.question,
                "answer": answer,
                "contexts_count": len(contexts),
                "chunk_ids": chunk_ids,
            })

            # Small delay to respect rate limits
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Sample {sample.sample_id} failed: {e}")
            per_sample.append({
                "sample_id": sample.sample_id,
                "error": str(e),
            })

    if not questions:
        logger.error(f"No valid samples for {strategy_name}")
        return StrategyResult(strategy=strategy_name)

    # Build RAGAS dataset
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # Run RAGAS evaluation
    logger.info(f"Running RAGAS metrics for {strategy_name}...")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    # Extract scores
    df = result.to_pandas()

    strategy_result = StrategyResult(
        strategy=strategy_name,
        context_precision=float(df["context_precision"].mean()),
        context_recall=float(df["context_recall"].mean()),
        faithfulness=float(df["faithfulness"].mean()),
        answer_relevancy=float(df["answer_relevancy"].mean()),
        per_sample=per_sample,
    )

    logger.info(
        f"{strategy_name} scores: "
        f"cp={strategy_result.context_precision:.3f} "
        f"cr={strategy_result.context_recall:.3f} "
        f"f={strategy_result.faithfulness:.3f} "
        f"ar={strategy_result.answer_relevancy:.3f} "
        f"avg={strategy_result.average:.3f}"
    )

    return strategy_result


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(
    subset: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
) -> RAGASReport:
    """
    Run full RAGAS evaluation across all three strategies.

    Args:
        subset:     list of sample_ids to run (None = all 28)
        categories: list of categories to run (None = all)

    Returns:
        RAGASReport with all results
    """
    if not RAGAS_AVAILABLE:
        raise ImportError(
            "RAGAS not available. Install with: "
            "pip install ragas langchain-mistralai datasets"
        )

    from src.retrieval.reranker import RetrievalPipeline

    # Load dataset
    dataset = get_dataset()
    if subset:
        dataset = [s for s in dataset if s.sample_id in subset]
    if categories:
        dataset = [s for s in dataset if s.category in categories]

    logger.info(f"Evaluation dataset: {len(dataset)} samples")

    # Initialise retrieval pipeline
    logger.info("Loading retrieval pipeline...")
    pipeline = RetrievalPipeline.from_env()

    # Initialise RAGAS LLM and embeddings (using Mistral)
    llm = LangchainLLMWrapper(
        ChatMistralAI(
            model=MISTRAL_LARGE_MODEL,
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.1,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(
            model=MISTRAL_EMBED_MODEL,
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        )
    )

    report = RAGASReport(
        dataset_size=len(dataset),
        evaluation_date=time.strftime("%Y-%m-%d"),
        model_used=MISTRAL_LARGE_MODEL,
        notes=f"Evaluated {len(dataset)} samples across 3 retrieval strategies",
    )

    # Strategy 1: Dense only
    dense_result = run_ragas_evaluation(
        strategy_name="dense",
        retrieval_fn=retrieve_dense,
        pipeline=pipeline,
        dataset=dataset,
        llm=llm,
        embeddings=embeddings,
    )
    report.strategies["dense"] = dense_result

    # Strategy 2: Hybrid (dense + sparse + RRF)
    hybrid_result = run_ragas_evaluation(
        strategy_name="hybrid",
        retrieval_fn=retrieve_hybrid,
        pipeline=pipeline,
        dataset=dataset,
        llm=llm,
        embeddings=embeddings,
    )
    report.strategies["hybrid"] = hybrid_result

    # Strategy 3: Reranked (hybrid + cross-encoder)
    reranked_result = run_ragas_evaluation(
        strategy_name="reranked",
        retrieval_fn=retrieve_reranked,
        pipeline=pipeline,
        dataset=dataset,
        llm=llm,
        embeddings=embeddings,
    )
    report.strategies["reranked"] = reranked_result

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {RESULTS_PATH}")
    logger.info(f"Winner: {report.winner()}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--subset", nargs="+", help="Sample IDs to evaluate")
    parser.add_argument("--categories", nargs="+", help="Categories to evaluate")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 8 samples only (one per category)")
    args = parser.parse_args()

    if args.quick:
        # One sample per category for a quick sanity check (~$0.30, ~5 min)
        quick_ids = ["cls_01", "cls_05", "obl_01", "obl_04",
                     "tra_01", "gpai_01", "xref_01", "xref_04"]
        print(f"Quick run: {len(quick_ids)} samples")
        report = run_full_evaluation(subset=quick_ids)
    else:
        report = run_full_evaluation(
            subset=args.subset,
            categories=args.categories,
        )

    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {report.dataset_size} samples")
    print(f"Date:    {report.evaluation_date}")
    print()

    metrics = ["context_precision", "context_recall", "faithfulness",
               "answer_relevancy", "average"]
    header = f"{'Metric':22s}" + "".join(f"{s:>12s}" for s in report.strategies)
    print(header)
    print("-" * len(header))

    for metric in metrics:
        row = f"{'  '+metric:22s}"
        for strategy in report.strategies.values():
            val = getattr(strategy, metric, strategy.average)
            row += f"{val:>12.3f}"
        if metric == "average":
            print("-" * len(header))
        print(row)

    print()
    print(f"Winner: {report.winner().upper()}")

    improvements = report.to_dict()["improvements"]
    print(f"\nImprovements:")
    print(f"  Hybrid vs Dense:    avg {improvements['hybrid_vs_dense']['average']:+.3f}")
    print(f"  Reranked vs Hybrid: avg {improvements['reranked_vs_hybrid']['average']:+.3f}")
    print(f"  Reranked vs Dense:  avg {improvements['reranked_vs_dense']['average']:+.3f}")
