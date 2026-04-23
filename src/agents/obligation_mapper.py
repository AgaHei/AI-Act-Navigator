"""
obligation_mapper.py — Maps risk tier to applicable AI Act obligations

Takes a ClassificationResult and retrieves the specific obligations
that apply to the system's risk tier and actor role.

This is where the agentic cross-reference resolution happens:
  - For each applicable article, follow its cross-references
  - Retrieve the referenced annexes automatically
  - Build a complete obligation picture, not just the top-level articles

Output: ObligationMap — structured list of obligations grouped by theme,
each with article reference, description, actor, and deadline.
"""

import logging
import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from mistralai import Mistral

from .classifier import ClassificationResult, RiskTier
from ..retrieval.reranker import RetrievalPipeline

load_dotenv()
logger = logging.getLogger(__name__)

MISTRAL_LARGE_MODEL = os.getenv("MISTRAL_LARGE_MODEL", "mistral-large-latest")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Obligation:
    """A single compliance obligation."""
    article_ref: str               # e.g. "Art. 9(1)"
    title: str                     # short title
    description: str               # plain language description
    actor: str                     # "provider" | "deployer" | "both"
    deadline: str                  # ISO date string
    is_mandatory: bool = True
    theme: str = "general"         # grouping theme
    annex_refs: list[str] = field(default_factory=list)  # related annexes

    def to_dict(self) -> dict:
        return {
            "article_ref": self.article_ref,
            "title": self.title,
            "description": self.description,
            "actor": self.actor,
            "deadline": self.deadline,
            "is_mandatory": self.is_mandatory,
            "theme": self.theme,
            "annex_refs": self.annex_refs,
        }


@dataclass
class ObligationMap:
    """
    Complete set of obligations applicable to a classified AI system.
    Grouped by theme for structured display.
    """
    risk_tier: RiskTier
    actor_role: str
    obligations: list[Obligation] = field(default_factory=list)
    retrieved_chunks: list = field(default_factory=list)
    mapping_reasoning: str = ""

    def by_theme(self) -> dict[str, list[Obligation]]:
        """Group obligations by theme."""
        grouped: dict[str, list[Obligation]] = {}
        for ob in self.obligations:
            grouped.setdefault(ob.theme, []).append(ob)
        return grouped

    def by_deadline(self) -> list[Obligation]:
        """Sort obligations by deadline ascending."""
        return sorted(self.obligations, key=lambda o: o.deadline)

    def mandatory_count(self) -> int:
        return sum(1 for o in self.obligations if o.is_mandatory)

    def summary(self) -> str:
        lines = [
            f"Risk tier:    {self.risk_tier.display()}",
            f"Actor role:   {self.actor_role}",
            f"Obligations:  {len(self.obligations)} "
            f"({self.mandatory_count()} mandatory)",
            "",
        ]
        for theme, obs in self.by_theme().items():
            lines.append(f"  {theme.upper()}:")
            for ob in obs:
                flag = "✅" if ob.is_mandatory else "○"
                lines.append(
                    f"    {flag} {ob.article_ref} — {ob.title} "
                    f"[{ob.actor}] deadline: {ob.deadline}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Obligation retrieval queries — per risk tier
# ---------------------------------------------------------------------------

TIER_QUERIES = {
    RiskTier.PROHIBITED: [
        ("Article 5 prohibited AI practices obligations", ["prohibited"]),
    ],
    RiskTier.HIGH_RISK: [
        ("Article 9 risk management system requirements provider", ["high_risk"]),
        ("Article 10 data governance training data high-risk", ["high_risk"]),
        ("Article 11 technical documentation high-risk AI", ["high_risk"]),
        ("Article 12 record-keeping logging high-risk AI", ["high_risk"]),
        ("Article 13 transparency information users high-risk", ["high_risk"]),
        ("Article 14 human oversight high-risk AI systems", ["high_risk"]),
        ("Article 15 accuracy robustness cybersecurity high-risk", ["high_risk"]),
        ("Article 16 obligations providers high-risk AI", ["high_risk"]),
        ("Article 17 quality management system provider", ["high_risk"]),
        ("Article 49 registration EU database high-risk", ["high_risk"]),
        ("Annex IV technical documentation requirements", ["high_risk"]),
    ],
    RiskTier.LIMITED_RISK: [
        ("Article 50 transparency obligations AI systems", ["limited_risk"]),
        ("chatbot transparency disclosure AI interaction", ["limited_risk"]),
    ],
    RiskTier.GPAI: [
        ("Article 53 transparency obligations GPAI providers", ["gpai"]),
        ("Article 54 copyright GPAI model training data", ["gpai"]),
        ("Article 55 systemic risk GPAI model obligations", ["gpai"]),
        ("Annex XI GPAI model technical documentation", ["gpai"]),
    ],
    RiskTier.MINIMAL_RISK: [
        ("voluntary codes of conduct minimal risk AI", ["all"]),
    ],
}

OBLIGATION_MAPPING_PROMPT = """You are an EU AI Act compliance expert.
Given a classified AI system and relevant AI Act provisions, identify ALL
applicable obligations for the specified actor role.

For each obligation provide:
- The specific article/paragraph reference
- A short title (5-8 words)
- Plain language description (1-2 sentences)
- Who it applies to (provider/deployer/both)
- The compliance deadline
- Theme category: one of [risk_management, data_governance, transparency,
  human_oversight, technical_documentation, conformity_assessment,
  registration, post_market, governance, voluntary]

IMPORTANT:
- Only include obligations that genuinely apply given the risk tier and actor
- For HIGH-RISK: provider obligations (Art. 16-27) are extensive
- For LIMITED-RISK: mainly Art. 50 transparency
- "Not yet deployed" providers still have Art. 16-22 obligations
- Flag obligations that may not apply depending on clarification

Respond with ONLY valid JSON:
{
  "obligations": [
    {
      "article_ref": "Art. X(Y)",
      "title": "short title",
      "description": "plain language description",
      "actor": "provider|deployer|both",
      "deadline": "YYYY-MM-DD",
      "is_mandatory": true/false,
      "theme": "theme_name",
      "annex_refs": ["Annex IV"]
    }
  ],
  "mapping_reasoning": "brief explanation of mapping decisions"
}"""


# ---------------------------------------------------------------------------
# Obligation mapper
# ---------------------------------------------------------------------------

class ObligationMapper:
    """
    Maps a classified AI system to its applicable obligations.

    Uses targeted retrieval per risk tier to pull the relevant
    obligation articles, then asks mistral-large to identify
    which obligations apply given the system's specific context.
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

    def map_obligations(
        self, classification: ClassificationResult
    ) -> ObligationMap:
        """
        Map applicable obligations for a classified AI system.

        Args:
            classification: output from RiskClassifier

        Returns:
            ObligationMap with all applicable obligations
        """
        tier = classification.risk_tier
        form = classification.intake_form
        actor = form.actor_role if form else "provider"

        logger.info(f"Mapping obligations for tier={tier.value}, actor={actor}")

        # Retrieve obligation chunks for this tier
        chunks = self._retrieve_obligation_chunks(tier)

        # Add transparency chunks if applicable
        if classification.transparency_obligations and tier != RiskTier.LIMITED_RISK:
            transp_results = self._retrieval.retrieve(
                "Article 50 transparency obligations chatbot",
                top_k=2,
            )
            seen_ids = {c.chunk_id for c in chunks}
            for r in transp_results:
                if r.chunk_id not in seen_ids:
                    chunks.append(r)

        # Build prompt and call LLM
        user_prompt = self._build_mapping_prompt(classification, chunks, actor)
        raw = self._call_mapper(user_prompt)
        obligation_map = self._parse_obligations(raw, tier, actor, chunks)

        logger.info(
            f"Mapped {len(obligation_map.obligations)} obligations "
            f"({obligation_map.mandatory_count()} mandatory)"
        )

        return obligation_map

    def _retrieve_obligation_chunks(self, tier: RiskTier) -> list:
        """Retrieve obligation chunks for the given risk tier."""
        queries = TIER_QUERIES.get(tier, [])
        seen_ids: set = set()
        chunks = []

        for query, tiers in queries:
            results = self._retrieval.retrieve(
                query=query,
                top_k=2,
                risk_tiers=tiers,
            )
            for r in results:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    chunks.append(r)

        return chunks[:15]  # cap at 15 chunks

    def _build_mapping_prompt(
        self,
        classification: ClassificationResult,
        chunks: list,
        actor: str,
    ) -> str:
        form = classification.intake_form
        system_desc = form.summary() if form else "System details unavailable"

        context = "\n".join([
            f"[{i+1}] {c.display_reference}\n{c.text[:350]}\n"
            for i, c in enumerate(chunks)
        ])

        return (
            f"CLASSIFIED AI SYSTEM:\n{system_desc}\n"
            f"Risk tier: {classification.risk_tier.display()}\n"
            f"Actor role: {actor}\n"
            f"Annex III domain: {classification.annex_iii_domain or 'N/A'}\n\n"
            f"RELEVANT AI ACT OBLIGATIONS:\n{context}\n\n"
            f"List all applicable obligations for this system and actor role."
        )

    def _call_mapper(self, user_prompt: str) -> str:
        response = self._client.chat.complete(
            model=MISTRAL_LARGE_MODEL,
            messages=[
                {"role": "system", "content": OBLIGATION_MAPPING_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def _parse_obligations(
        self,
        raw: str,
        tier: RiskTier,
        actor: str,
        chunks: list,
    ) -> ObligationMap:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Obligation parse error: {e}")
            return ObligationMap(
                risk_tier=tier,
                actor_role=actor,
                retrieved_chunks=chunks,
                mapping_reasoning="Parse error",
            )

        obligations = []
        for ob_data in data.get("obligations", []):
            obligations.append(Obligation(
                article_ref=ob_data.get("article_ref", ""),
                title=ob_data.get("title", ""),
                description=ob_data.get("description", ""),
                actor=ob_data.get("actor", "provider"),
                deadline=ob_data.get("deadline", "2026-08-02"),
                is_mandatory=ob_data.get("is_mandatory", True),
                theme=ob_data.get("theme", "general"),
                annex_refs=ob_data.get("annex_refs", []),
            ))

        return ObligationMap(
            risk_tier=tier,
            actor_role=actor,
            obligations=obligations,
            retrieved_chunks=chunks,
            mapping_reasoning=data.get("mapping_reasoning", ""),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_obligation_mapper(
    collection_name: str = "ai_act_navigator",
) -> ObligationMapper:
    pipeline = RetrievalPipeline.from_env(collection_name=collection_name)
    return ObligationMapper(retrieval_pipeline=pipeline)
