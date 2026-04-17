"""
extractor.py — Free-text to structured intake form for AI Act Navigator

Takes a natural language description of an AI system and extracts
structured attributes needed for risk classification.

Pipeline:
  Free-text description
        ↓
  LLM (mistral-small) → structured JSON + confidence scores per field
        ↓
  Confidence gate:
    - All critical fields ≥ threshold → proceed to classifier
    - Any critical field < threshold  → generate targeted follow-up question
        ↓ (after ≤ MAX_CLARIFICATION_ROUNDS)
  IntakeForm (dataclass) — validated, ready for classifier

Design decisions:
  - mistral-small for extraction (cheap, fast — structured output task)
  - Confidence scoring per field — enables surgical clarification
  - Max 2 clarification rounds — prevents interrogation UX
  - Uncertain fields flagged in output — transparency about confidence
  - JSON output with schema enforcement via system prompt
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MISTRAL_SMALL_MODEL = os.getenv("MISTRAL_SMALL_MODEL", "open-mistral-7b")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MAX_CLARIFICATION_ROUNDS = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "2"))

# Critical fields — below threshold these trigger clarification
CRITICAL_FIELDS = {"sector", "user_base", "autonomy_level", "deployment_status"}


# ---------------------------------------------------------------------------
# Data model — the structured intake form
# ---------------------------------------------------------------------------

@dataclass
class IntakeForm:
    """
    Structured representation of an AI system for risk classification.
    Populated by the LLM extractor from free-text description.
    """
    # Core classification fields
    system_name: Optional[str] = None
    system_description: str = ""

    # Sector — maps to Annex III categories
    # e.g. "education", "employment", "healthcare", "law_enforcement",
    #      "critical_infrastructure", "migration", "justice", "other"
    sector: Optional[str] = None

    # User base characteristics
    # e.g. "general_public", "professionals", "minors", "vulnerable_groups",
    #      "employees", "students"
    user_base: Optional[str] = None

    # Whether the system makes autonomous decisions affecting users
    # "autonomous" | "advisory" | "informational"
    autonomy_level: Optional[str] = None

    # Deployment status — affects provider/deployer distinction
    # "deployed_public" | "deployed_restricted" | "internal_use" |
    # "not_deployed" | "prototype"
    deployment_status: Optional[str] = None

    # Data types processed
    # e.g. ["text", "biometric", "health", "financial", "location"]
    data_types: list[str] = field(default_factory=list)

    # Actor role of the person/org describing the system
    # "provider" | "deployer" | "both" | "unknown"
    actor_role: Optional[str] = None

    # GPAI model dependency
    # Name of foundation model used, if any (e.g. "OpenAI GPT-4", "Mistral")
    gpai_model: Optional[str] = None

    # EU market presence
    # "yes" | "no" | "unknown"
    targets_eu_market: str = "unknown"

    # Confidence scores per field (0.0 → 1.0)
    confidence: dict = field(default_factory=dict)

    # Fields below confidence threshold — populated by extractor
    uncertain_fields: list[str] = field(default_factory=list)

    # Raw description for reference
    raw_description: str = ""

    def is_ready_for_classification(self) -> bool:
        """True if all critical fields meet confidence threshold."""
        for f in CRITICAL_FIELDS:
            if self.confidence.get(f, 0.0) < CONFIDENCE_THRESHOLD:
                return False
        return True

    def get_uncertain_critical_fields(self) -> list[str]:
        """Return critical fields below confidence threshold."""
        return [
            f for f in CRITICAL_FIELDS
            if self.confidence.get(f, 0.0) < CONFIDENCE_THRESHOLD
        ]

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable summary for display."""
        lines = [
            f"System:      {self.system_name or 'unnamed'}",
            f"Sector:      {self.sector or '?'} (conf: {self.confidence.get('sector', 0):.0%})",
            f"User base:   {self.user_base or '?'} (conf: {self.confidence.get('user_base', 0):.0%})",
            f"Autonomy:    {self.autonomy_level or '?'} (conf: {self.confidence.get('autonomy_level', 0):.0%})",
            f"Deployment:  {self.deployment_status or '?'} (conf: {self.confidence.get('deployment_status', 0):.0%})",
            f"Data types:  {', '.join(self.data_types) or '?'}",
            f"Actor role:  {self.actor_role or '?'}",
            f"GPAI model:  {self.gpai_model or 'none/unknown'}",
            f"EU market:   {self.targets_eu_market}",
        ]
        if self.uncertain_fields:
            lines.append(f"\n⚠ Uncertain fields: {', '.join(self.uncertain_fields)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt for extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are an expert AI Act compliance analyst.
Your task is to extract structured information from a description of an AI system.

You MUST respond with ONLY a valid JSON object — no preamble, no explanation,
no markdown code blocks. Raw JSON only.

The JSON must follow this exact schema:
{
  "system_name": "string or null",
  "sector": "one of: education, employment, healthcare, law_enforcement, critical_infrastructure, migration, justice, biometric, other",
  "user_base": "one of: general_public, professionals, minors, vulnerable_groups, employees, students, mixed",
  "autonomy_level": "one of: autonomous, advisory, informational",
  "deployment_status": "one of: deployed_public, deployed_restricted, internal_use, not_deployed, prototype",
  "data_types": ["array of: text, biometric, health, financial, location, behavioral, other"],
  "actor_role": "one of: provider, deployer, both, unknown",
  "gpai_model": "name of foundation model used, or null",
  "targets_eu_market": "one of: yes, no, unknown",
  "confidence": {
    "sector": 0.0-1.0,
    "user_base": 0.0-1.0,
    "autonomy_level": 0.0-1.0,
    "deployment_status": 0.0-1.0,
    "data_types": 0.0-1.0,
    "actor_role": 0.0-1.0,
    "gpai_model": 0.0-1.0,
    "targets_eu_market": 0.0-1.0
  },
  "reasoning": "brief explanation of key classification decisions"
}

Confidence guidelines:
- 0.9-1.0: explicitly stated in description
- 0.7-0.9: strongly implied
- 0.5-0.7: uncertain, could go either way
- 0.0-0.5: mostly guessing, needs clarification

For sector: choose the PRIMARY sector even if multiple apply.
For user_base: if vulnerable groups (minors, people with disabilities,
  mental health conditions) are mentioned, always choose vulnerable_groups.
For autonomy_level:
  - autonomous: system makes decisions without human review
  - advisory: system provides recommendations, human decides
  - informational: system provides information only, no decision support
For actor_role: provider = builds/trains the system,
  deployer = uses an existing system in their application."""


CLARIFICATION_SYSTEM_PROMPT = """You are an expert AI Act compliance analyst.
You need to ask ONE targeted clarification question to resolve uncertainty
about a specific field in an AI system description.

The question must:
- Be specific to the uncertain field
- Offer 2-3 concrete answer options where appropriate
- Be phrased professionally but accessibly
- NOT ask about other fields

Respond with ONLY a JSON object:
{
  "field": "the uncertain field name",
  "question": "your clarification question",
  "options": ["option 1", "option 2", "option 3"]
}"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class AISystemExtractor:
    """
    Extracts structured intake form from free-text AI system description.

    Uses mistral-small for cost efficiency — extraction is a structured
    output task that doesn't require the reasoning power of mistral-large.

    Implements a confidence-gated clarification loop:
    1. Extract structured fields + confidence scores
    2. Check if critical fields meet threshold
    3. If not: generate targeted clarification question
    4. Update form with clarification response
    5. Repeat up to MAX_CLARIFICATION_ROUNDS
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise ValueError("MISTRAL_API_KEY not set")
        self._client = Mistral(api_key=key)

    def extract(self, description: str) -> IntakeForm:
        """
        Extract structured form from description (single pass, no clarification).
        Use extract_with_clarification() for the full interactive pipeline.
        """
        raw = self._call_extraction(description)
        form = self._parse_extraction(raw, description)
        return form

    def extract_with_clarification(
        self,
        description: str,
        clarification_callback=None,
    ) -> IntakeForm:
        """
        Full extraction pipeline with clarification loop.

        Args:
            description:             initial free-text description
            clarification_callback:  function(question: str, options: list) → str
                                     receives the clarification question,
                                     returns the user's answer as a string.
                                     If None, proceeds without clarification.

        Returns:
            IntakeForm populated with best available information
        """
        # Initial extraction
        form = self.extract(description)
        logger.info(f"Initial extraction complete. Ready: {form.is_ready_for_classification()}")

        if form.is_ready_for_classification() or clarification_callback is None:
            return form

        # Clarification loop
        conversation = [description]
        rounds = 0

        while (
            not form.is_ready_for_classification()
            and rounds < MAX_CLARIFICATION_ROUNDS
        ):
            uncertain = form.get_uncertain_critical_fields()
            if not uncertain:
                break

            # Generate clarification for most uncertain field
            target_field = min(uncertain, key=lambda f: form.confidence.get(f, 0.0))
            question_data = self._generate_clarification(
                description="\n".join(conversation),
                uncertain_field=target_field,
                current_value=getattr(form, target_field),
                current_confidence=form.confidence.get(target_field, 0.0),
            )

            # Ask user
            logger.info(f"Requesting clarification on: {target_field}")
            answer = clarification_callback(
                question_data.get("question", f"Please clarify: {target_field}"),
                question_data.get("options", []),
            )

            # Re-extract with clarification context
            conversation.append(
                f"Clarification on {target_field}: {answer}"
            )
            full_context = (
                f"Original description: {conversation[0]}\n\n"
                f"Additional context provided:\n" +
                "\n".join(conversation[1:])
            )
            form = self.extract(full_context)
            rounds += 1
            logger.info(
                f"Round {rounds}: confidence after clarification — "
                f"{target_field}: {form.confidence.get(target_field, 0):.0%}"
            )

        # Flag remaining uncertain fields in the form
        still_uncertain = form.get_uncertain_critical_fields()
        if still_uncertain:
            form.uncertain_fields = still_uncertain
            logger.warning(
                f"Proceeding with uncertain fields after "
                f"{rounds} clarification rounds: {still_uncertain}"
            )

        return form

    def _call_extraction(self, description: str) -> str:
        """Call mistral-small for structured extraction."""
        response = self._client.chat.complete(
            model=MISTRAL_SMALL_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract structured information from this AI system description:\n\n{description}"},
            ],
            temperature=0.1,   # low temperature for consistent structured output
            max_tokens=800,
        )
        return response.choices[0].message.content

    def _parse_extraction(self, raw: str, original_description: str) -> IntakeForm:
        """Parse LLM JSON output into IntakeForm."""
        # Strip markdown code blocks if present
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw output: {raw[:300]}")
            # Return minimal form with zero confidence
            return IntakeForm(
                raw_description=original_description,
                system_description=original_description,
                confidence={f: 0.0 for f in CRITICAL_FIELDS},
            )

        confidence = data.get("confidence", {})

        form = IntakeForm(
            system_name=data.get("system_name"),
            system_description=original_description,
            raw_description=original_description,
            sector=data.get("sector"),
            user_base=data.get("user_base"),
            autonomy_level=data.get("autonomy_level"),
            deployment_status=data.get("deployment_status"),
            data_types=data.get("data_types", []),
            actor_role=data.get("actor_role"),
            gpai_model=data.get("gpai_model"),
            targets_eu_market=data.get("targets_eu_market", "unknown"),
            confidence=confidence,
            uncertain_fields=[
                f for f in CRITICAL_FIELDS
                if confidence.get(f, 0.0) < CONFIDENCE_THRESHOLD
            ],
        )

        # Log reasoning if present
        if reasoning := data.get("reasoning"):
            logger.info(f"Extractor reasoning: {reasoning}")

        return form

    def _generate_clarification(
        self,
        description: str,
        uncertain_field: str,
        current_value: Optional[str],
        current_confidence: float,
    ) -> dict:
        """Generate a targeted clarification question for an uncertain field."""
        user_msg = (
            f"AI system description so far:\n{description}\n\n"
            f"I need clarification on the '{uncertain_field}' field.\n"
            f"Current best guess: {current_value!r} (confidence: {current_confidence:.0%})\n"
            f"Please generate ONE targeted question to resolve this uncertainty."
        )

        response = self._client.chat.complete(
            model=MISTRAL_SMALL_MODEL,
            messages=[
                {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=300,
        )

        raw = response.choices[0].message.content
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "field": uncertain_field,
                "question": f"Could you clarify the {uncertain_field} of your system?",
                "options": [],
            }


# ---------------------------------------------------------------------------
# CLI — test with Complice description
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    extractor = AISystemExtractor()

    # The real Complice description
    complice_description = (
        "Complice is a RAG-based conversational assistant for young adults "
        "with ASD aged 16-25, not deployed on a public web platform, built "
        "on OpenAI, offering emotional support and information on mental "
        "health topics."
    )

    print("="*65)
    print("AI Act Navigator — Extractor Test")
    print("="*65)
    print(f"\nInput description:\n{complice_description}\n")
    print("Extracting structured form...\n")

    form = extractor.extract(complice_description)

    print("="*65)
    print("EXTRACTED INTAKE FORM")
    print("="*65)
    print(form.summary())

    print("\n" + "="*65)
    print("CONFIDENCE SCORES")
    print("="*65)
    for field_name, score in form.confidence.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        flag = " ⚠" if score < CONFIDENCE_THRESHOLD else " ✅"
        print(f"  {field_name:20s} {bar} {score:.0%}{flag}")

    print("\n" + "="*65)
    print("CLASSIFICATION READINESS")
    print("="*65)
    if form.is_ready_for_classification():
        print("✅ Ready for classification — all critical fields above threshold")
    else:
        print("⚠  Clarification needed for:")
        for f in form.get_uncertain_critical_fields():
            print(f"   - {f} (confidence: {form.confidence.get(f, 0):.0%})")

    print("\n" + "="*65)
    print("RAW JSON OUTPUT")
    print("="*65)
    print(json.dumps(form.to_dict(), indent=2, ensure_ascii=False))
