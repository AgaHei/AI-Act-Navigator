"""
classifier.py — Risk tier classification agent for AI Act Navigator

Takes a populated IntakeForm and determines the AI Act risk tier
through structured legal reasoning grounded in retrieved corpus chunks.

Classification logic follows the AI Act's own decision tree:
  Step 1 — Is the practice prohibited? (Art. 5)
  Step 2 — Is the system high-risk? (Art. 6 + Annex III)
  Step 3 — Does it trigger transparency obligations? (Art. 50)
  Step 4 — Is it a GPAI model? (Art. 51-56)
  Step 5 — Minimal risk (no specific obligations)

Uses mistral-large for reasoning — classification has legal consequences
and requires the strongest available model.

Each classification decision is:
  - Grounded in retrieved corpus chunks (not LLM hallucination)
  - Accompanied by article references
  - Assigned a confidence score
  - Explained in plain language for the consultant
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from mistralai.client import Mistral

from .extractor import IntakeForm
from ..retrieval.reranker import RetrievalPipeline

load_dotenv()
logger = logging.getLogger(__name__)

MISTRAL_LARGE_MODEL = os.getenv("MISTRAL_LARGE_MODEL", "mistral-large-latest")


# ---------------------------------------------------------------------------
# Risk tier enum
# ---------------------------------------------------------------------------

class RiskTier(str, Enum):
    PROHIBITED      = "prohibited"
    HIGH_RISK       = "high_risk"
    LIMITED_RISK    = "limited_risk"
    GPAI            = "gpai"
    MINIMAL_RISK    = "minimal_risk"
    UNCERTAIN       = "uncertain"

    def display(self) -> str:
        labels = {
            "prohibited":   "🚫 Prohibited",
            "high_risk":    "🔴 High Risk",
            "limited_risk": "🟡 Limited Risk",
            "gpai":         "🟣 GPAI Model",
            "minimal_risk": "🟢 Minimal Risk",
            "uncertain":    "❓ Uncertain",
        }
        return labels.get(self.value, self.value)

    def deadline(self) -> str:
        deadlines = {
            "prohibited":   "2025-02-02 (already in force)",
            "high_risk":    "2026-08-02",
            "limited_risk": "2026-08-02",
            "gpai":         "2025-08-02 (already in force)",
            "minimal_risk": "no mandatory deadline",
            "uncertain":    "assessment required",
        }
        return deadlines.get(self.value, "unknown")


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """
    The output of the classification agent.
    Contains the risk tier determination with full legal reasoning.
    """
    # Primary classification
    risk_tier: RiskTier
    confidence: float               # 0.0 → 1.0

    # Legal grounding
    primary_articles: list[str]     # e.g. ["Art. 6(2)", "Annex III point 4"]
    supporting_articles: list[str]  # secondary references
    retrieved_chunks: list          # RetrievedChunk objects used for reasoning

    # Reasoning chain
    reasoning: str                  # full legal reasoning in plain language
    key_factors: list[str]          # bullet points driving the decision
    borderline_considerations: list[str]  # edge cases / precautionary notes

    # Annex III specific (if high-risk)
    annex_iii_domain: Optional[str] = None
    annex_iii_point: Optional[str] = None

    # Art. 50 transparency (may apply regardless of tier)
    transparency_obligations: bool = False
    transparency_reasoning: Optional[str] = None

    # GPAI model dependency
    gpai_dependency_noted: bool = False
    gpai_dependency_reasoning: Optional[str] = None

    # Metadata
    intake_form: Optional[IntakeForm] = None
    model_used: str = MISTRAL_LARGE_MODEL

    def summary(self) -> str:
        lines = [
            f"Risk tier:    {self.risk_tier.display()}",
            f"Confidence:   {self.confidence:.0%}",
            f"Deadline:     {self.risk_tier.deadline()}",
            f"",
            f"Primary articles: {', '.join(self.primary_articles)}",
            f"",
            f"Key factors:",
        ]
        for factor in self.key_factors:
            lines.append(f"  • {factor}")

        if self.borderline_considerations:
            lines.append("")
            lines.append("Borderline considerations:")
            for note in self.borderline_considerations:
                lines.append(f"  ⚠  {note}")

        if self.transparency_obligations:
            lines.append("")
            lines.append(f"📋 Art. 50 transparency obligations apply")
            if self.transparency_reasoning:
                lines.append(f"   {self.transparency_reasoning}")

        if self.gpai_dependency_noted:
            lines.append("")
            lines.append(f"🔗 GPAI model dependency noted")
            if self.gpai_dependency_reasoning:
                lines.append(f"   {self.gpai_dependency_reasoning}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt for classification
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM_PROMPT = """You are an expert EU AI Act compliance lawyer.
Your task is to classify an AI system into the correct risk tier under
Regulation (EU) 2024/1689 (the AI Act).

You will be given:
1. A structured description of the AI system (IntakeForm)
2. Relevant excerpts from the AI Act retrieved by a RAG system

Follow the AI Act's decision tree STRICTLY:

STEP 1 — PROHIBITED (Art. 5)?
Check: subliminal manipulation, exploitation of vulnerabilities,
social scoring by public authorities, real-time biometric ID in public spaces,
emotion recognition in workplace/education, AI-generated deepfakes without disclosure.
If YES → PROHIBITED. Stop.

STEP 2 — HIGH-RISK (Art. 6 + Annex III)?
Check Annex III categories:
  1. Biometric identification
  2. Critical infrastructure
  3. Education and vocational training
  4. Employment and workers management
  5. Essential private/public services (credit, social benefits)
  6. Law enforcement
  7. Migration and asylum
  8. Administration of justice
Also check Art. 6(1): AI in safety-critical products (Annex II).
If YES → HIGH-RISK. Identify the specific Annex III domain.
EXCEPTION: Art. 6(3) — provider may self-assess as NOT high-risk if
  the system does not pose significant risk. Must be documented.

STEP 3 — LIMITED RISK (Art. 50)?
Check: chatbots, emotion recognition systems, deep fake generators,
AI systems generating synthetic content.
If YES → LIMITED RISK with transparency obligations.

STEP 4 — GPAI MODEL (Art. 51-56)?
Check: is this a general-purpose AI model (not a downstream application)?
Note: downstream deployers using a GPAI model are NOT covered by Title V.
If the system IS a GPAI model → GPAI tier.

STEP 5 → MINIMAL RISK (no specific mandatory obligations).

IMPORTANT NOTES:
- Multiple tiers can apply (e.g. high-risk + Art. 50 transparency)
- "Not yet deployed" does NOT exempt from provider obligations
- Vulnerable users (minors, disabled, mental health) trigger higher scrutiny
- When uncertain between HIGH-RISK and LIMITED-RISK, flag as borderline

You MUST respond with ONLY a valid JSON object:
{
  "risk_tier": "prohibited|high_risk|limited_risk|gpai|minimal_risk|uncertain",
  "confidence": 0.0-1.0,
  "primary_articles": ["Art. X", "Annex III point Y"],
  "supporting_articles": ["Art. X", "Art. Y"],
  "reasoning": "full legal reasoning paragraph",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "borderline_considerations": ["note 1", "note 2"],
  "annex_iii_domain": "domain name or null",
  "annex_iii_point": "point number or null",
  "transparency_obligations": true/false,
  "transparency_reasoning": "why Art. 50 applies or null",
  "gpai_dependency_noted": true/false,
  "gpai_dependency_reasoning": "note on GPAI model use or null"
}"""


# ---------------------------------------------------------------------------
# Classifier agent
# ---------------------------------------------------------------------------

class RiskClassifier:
    """
    Classifies an AI system into an AI Act risk tier.

    Process:
    1. Retrieve relevant corpus chunks based on intake form attributes
    2. Build classification prompt with retrieved context
    3. Call mistral-large for legal reasoning
    4. Parse and validate the classification result
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        api_key: Optional[str] = None,
    ):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise ValueError("MISTRAL_API_KEY not set")
        self._client = Mistral(api_key=key)
        self._retrieval = retrieval_pipeline

    def classify(self, form: IntakeForm) -> ClassificationResult:
        """
        Classify an AI system from its intake form.

        Args:
            form: populated IntakeForm from extractor

        Returns:
            ClassificationResult with tier, reasoning, and legal references
        """
        logger.info(f"Classifying: {form.system_name or 'unnamed system'}")

        # Step 1 — retrieve relevant chunks
        chunks = self._retrieve_classification_context(form)
        logger.info(f"Retrieved {len(chunks)} chunks for classification context")

        # Step 2 — build prompt with context
        user_prompt = self._build_classification_prompt(form, chunks)

        # Step 3 — call LLM
        raw = self._call_classifier(user_prompt)

        # Step 4 — parse result
        result = self._parse_result(raw, chunks, form)
        logger.info(
            f"Classification: {result.risk_tier.display()} "
            f"(confidence: {result.confidence:.0%})"
        )

        return result

    def _retrieve_classification_context(
        self, form: IntakeForm
    ) -> list:
        """
        Retrieve corpus chunks most relevant to classifying this system.

        Runs multiple targeted queries to ensure all relevant provisions
        are in context for the LLM:
        - Prohibition check (Art. 5)
        - High-risk classification (Art. 6 + Annex III for the sector)
        - Transparency obligations (Art. 50)
        - GPAI context if relevant
        """
        queries = []

        # Always check prohibited practices
        queries.append(("prohibited AI practices manipulation", None))

        # Sector-specific Annex III query
        if form.sector:
            queries.append((
                f"Annex III high-risk AI {form.sector} sector obligations",
                ["high_risk"],
            ))

        # Vulnerable user base → extra scrutiny
        if form.user_base in ("vulnerable_groups", "minors"):
            queries.append((
                f"AI systems vulnerable users {form.user_base} safeguards",
                ["high_risk", "limited_risk"],
            ))

        # Classification rules
        queries.append(("Article 6 classification rules high-risk AI systems", None))

        # Transparency if conversational
        if form.autonomy_level == "informational" or form.data_types and "text" in form.data_types:
            queries.append(("Article 50 transparency obligations chatbot", None))

        # GPAI if applicable
        if form.gpai_model:
            queries.append(("GPAI model downstream deployer obligations", ["gpai"]))

        # Collect unique chunks (deduplicate by chunk_id)
        seen_ids = set()
        all_chunks = []
        for query, tiers in queries:
            results = self._retrieval.retrieve(
                query=query,
                top_k=3,
                risk_tiers=tiers,
            )
            for r in results:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    all_chunks.append(r)

        return all_chunks[:12]  # cap at 12 chunks to stay within context window

    def _build_classification_prompt(
        self, form: IntakeForm, chunks: list
    ) -> str:
        """Build the classification prompt with intake form + retrieved context."""

        # Format intake form
        form_section = f"""AI SYSTEM TO CLASSIFY:
{form.summary()}
"""

        # Format retrieved chunks as legal context
        context_section = "RELEVANT AI ACT PROVISIONS (retrieved by RAG):\n"
        for i, chunk in enumerate(chunks, 1):
            context_section += (
                f"\n[{i}] {chunk.display_reference}\n"
                f"{chunk.text[:400]}\n"
                f"...\n"
            )

        return f"{form_section}\n{context_section}\n\nClassify this AI system."

    def _call_classifier(self, user_prompt: str) -> str:
        """Call mistral-large for classification reasoning."""
        response = self._client.chat.complete(
            model=MISTRAL_LARGE_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        return response.choices[0].message.content

    def _parse_result(
        self, raw: str, chunks: list, form: IntakeForm
    ) -> ClassificationResult:
        """Parse LLM JSON output into ClassificationResult."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw: {raw[:300]}")
            return ClassificationResult(
                risk_tier=RiskTier.UNCERTAIN,
                confidence=0.0,
                primary_articles=[],
                supporting_articles=[],
                retrieved_chunks=chunks,
                reasoning="Classification failed — JSON parse error",
                key_factors=["Parse error — please retry"],
                borderline_considerations=[],
                intake_form=form,
            )

        try:
            tier = RiskTier(data.get("risk_tier", "uncertain"))
        except ValueError:
            tier = RiskTier.UNCERTAIN

        return ClassificationResult(
            risk_tier=tier,
            confidence=float(data.get("confidence", 0.5)),
            primary_articles=data.get("primary_articles", []),
            supporting_articles=data.get("supporting_articles", []),
            retrieved_chunks=chunks,
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            borderline_considerations=data.get("borderline_considerations", []),
            annex_iii_domain=data.get("annex_iii_domain"),
            annex_iii_point=data.get("annex_iii_point"),
            transparency_obligations=data.get("transparency_obligations", False),
            transparency_reasoning=data.get("transparency_reasoning"),
            gpai_dependency_noted=data.get("gpai_dependency_noted", False),
            gpai_dependency_reasoning=data.get("gpai_dependency_reasoning"),
            intake_form=form,
            model_used=MISTRAL_LARGE_MODEL,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_classifier(
    collection_name: str = "ai_act_navigator",
) -> RiskClassifier:
    """Build a RiskClassifier from environment variables."""
    pipeline = RetrievalPipeline.from_env(collection_name=collection_name)
    return RiskClassifier(retrieval_pipeline=pipeline)


# ---------------------------------------------------------------------------
# CLI — test with Complice
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from .extractor import AISystemExtractor

    complice_description = (
        "Complice is a RAG-based conversational assistant for young adults "
        "with ASD aged 16-25, not deployed on a public web platform, built "
        "on OpenAI, offering emotional support and information on mental "
        "health topics."
    )

    print("="*65)
    print("AI Act Navigator — Classification Test")
    print("="*65)

    # Extract
    print("\nStep 1: Extracting intake form...")
    extractor = AISystemExtractor()
    form = extractor.extract(complice_description)
    print(form.summary())

    # Classify
    print("\nStep 2: Classifying risk tier...")
    classifier = get_classifier()
    result = classifier.classify(form)

    print("\n" + "="*65)
    print("CLASSIFICATION RESULT")
    print("="*65)
    print(result.summary())

    print("\n" + "="*65)
    print("FULL LEGAL REASONING")
    print("="*65)
    print(result.reasoning)

    print("\n" + "="*65)
    print("RETRIEVED CONTEXT USED")
    print("="*65)
    for i, chunk in enumerate(result.retrieved_chunks[:5], 1):
        print(f"  [{i}] {chunk.display_reference} (score: {chunk.score:.4f})")
