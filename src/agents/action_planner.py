"""
action_planner.py — Compliance action plan generator for AI Act Navigator

Takes an ObligationMap and generates a prioritised, actionable compliance
checklist with concrete steps, owners, and deadlines.

This is the final agent in the pipeline — the output that a consultant
actually hands to a client. It must be:
  - Concrete (specific actions, not vague recommendations)
  - Prioritised (most urgent first, by deadline and complexity)
  - Contextualised (adapted to the specific system and actor role)
  - Anchored (every action references the legal source)

Output: ComplianceReport — the full deliverable including classification,
obligations, and action plan in structured form.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from mistralai.client import Mistral

from .classifier import ClassificationResult, RiskTier
from .obligation_mapper import ObligationMap, Obligation

load_dotenv()
logger = logging.getLogger(__name__)

MISTRAL_LARGE_MODEL = os.getenv("MISTRAL_LARGE_MODEL", "mistral-large-latest")
TODAY = date.today().isoformat()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """A single compliance action step."""
    priority: int                  # 1 = highest
    title: str                     # short action title
    description: str               # what to do, concretely
    owner: str                     # "provider" | "deployer" | "legal team" | "technical team"
    deadline: str                  # ISO date
    effort: str                    # "low" | "medium" | "high"
    legal_anchor: str              # article reference
    status: str = "todo"           # "todo" | "in_progress" | "done"
    dependencies: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_overdue(self) -> bool:
        try:
            return self.deadline < TODAY
        except Exception:
            return False

    @property
    def urgency_label(self) -> str:
        if self.is_overdue:
            return "🚨 OVERDUE"
        if self.deadline <= "2025-12-31":
            return "🔴 URGENT"
        if self.deadline <= "2026-08-02":
            return "🟡 HIGH"
        return "🟢 STANDARD"

    def to_dict(self) -> dict:
        return {
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "owner": self.owner,
            "deadline": self.deadline,
            "effort": self.effort,
            "legal_anchor": self.legal_anchor,
            "status": self.status,
            "urgency": self.urgency_label,
            "notes": self.notes,
        }


@dataclass
class ComplianceReport:
    """
    The complete compliance assessment report — the final deliverable.

    Contains:
    - Executive summary
    - Risk classification with reasoning
    - Full obligation map
    - Prioritised action plan
    - Metadata
    """
    # System info
    system_name: str
    system_description: str
    assessment_date: str = field(default_factory=lambda: date.today().isoformat())

    # Pipeline outputs
    classification: Optional[ClassificationResult] = None
    obligation_map: Optional[ObligationMap] = None
    actions: list[Action] = field(default_factory=list)

    # Summary
    executive_summary: str = ""
    key_risks: list[str] = field(default_factory=list)
    immediate_actions: list[str] = field(default_factory=list)

    def risk_tier(self) -> RiskTier:
        return self.classification.risk_tier if self.classification else RiskTier.UNCERTAIN

    def action_count(self) -> int:
        return len(self.actions)

    def urgent_actions(self) -> list[Action]:
        return [a for a in self.actions if a.deadline <= "2026-08-02"]

    def full_summary(self) -> str:
        lines = [
            "=" * 65,
            f"AI ACT COMPLIANCE ASSESSMENT REPORT",
            f"System: {self.system_name}",
            f"Date:   {self.assessment_date}",
            "=" * 65,
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            self.executive_summary,
            "",
            "RISK CLASSIFICATION",
            "-" * 40,
        ]

        if self.classification:
            lines.append(self.classification.summary())

        lines += [
            "",
            "COMPLIANCE ACTION PLAN",
            "-" * 40,
            f"Total actions: {self.action_count()} "
            f"| Urgent (before Aug 2026): {len(self.urgent_actions())}",
            "",
        ]

        for action in sorted(self.actions, key=lambda a: (a.deadline, a.priority)):
            lines += [
                f"[{action.priority}] {action.urgency_label} {action.title}",
                f"     {action.description}",
                f"     Owner: {action.owner} | Effort: {action.effort} | "
                f"Deadline: {action.deadline} | Ref: {action.legal_anchor}",
                "",
            ]

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "system_name": self.system_name,
            "system_description": self.system_description,
            "assessment_date": self.assessment_date,
            "risk_tier": self.risk_tier().value,
            "risk_tier_display": self.risk_tier().display(),
            "executive_summary": self.executive_summary,
            "key_risks": self.key_risks,
            "immediate_actions": self.immediate_actions,
            "classification": {
                "confidence": self.classification.confidence if self.classification else 0,
                "primary_articles": self.classification.primary_articles if self.classification else [],
                "reasoning": self.classification.reasoning if self.classification else "",
                "key_factors": self.classification.key_factors if self.classification else [],
                "borderline_considerations": self.classification.borderline_considerations if self.classification else [],
                "transparency_obligations": self.classification.transparency_obligations if self.classification else False,
                "annex_iii_domain": self.classification.annex_iii_domain if self.classification else None,
            },
            "obligations": [
                ob.to_dict()
                for ob in (self.obligation_map.obligations if self.obligation_map else [])
            ],
            "actions": [a.to_dict() for a in self.actions],
        }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ACTION_PLAN_PROMPT = """You are an EU AI Act compliance consultant.
Generate a concrete, prioritised compliance action plan for an AI system
based on its classification and applicable obligations.

Each action must be:
- SPECIFIC: what exactly to do, not "ensure compliance with"
- CONCRETE: a real deliverable (document, process, test, registration)
- PRIORITISED: by deadline urgency and logical dependencies
- ANCHORED: reference the specific article

Generate 5-10 actions maximum. Focus on what the actor must do NOW
(already overdue or urgent) vs what can be planned for later.

Also generate:
- An executive summary (3-4 sentences) for a non-technical audience
- Key risks if non-compliant
- Top 3 immediate actions (the most urgent)

Respond with ONLY valid JSON:
{
  "executive_summary": "3-4 sentences for non-technical audience",
  "key_risks": ["risk if non-compliant 1", "risk 2", "risk 3"],
  "immediate_actions": ["top action 1", "top action 2", "top action 3"],
  "actions": [
    {
      "priority": 1,
      "title": "short action title",
      "description": "concrete what-to-do description",
      "owner": "provider|deployer|legal team|technical team",
      "deadline": "YYYY-MM-DD",
      "effort": "low|medium|high",
      "legal_anchor": "Art. X(Y)",
      "notes": "optional context"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Action planner
# ---------------------------------------------------------------------------

class ActionPlanner:
    """
    Generates a prioritised compliance action plan from an ObligationMap.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise ValueError("MISTRAL_API_KEY not set")
        self._client = Mistral(api_key=key)

    def plan(
        self,
        classification: ClassificationResult,
        obligation_map: ObligationMap,
    ) -> ComplianceReport:
        """
        Generate a compliance report with prioritised action plan.

        Args:
            classification:  output from RiskClassifier
            obligation_map:  output from ObligationMapper

        Returns:
            ComplianceReport — the complete deliverable
        """
        form = classification.intake_form
        system_name = (form.system_name if form else None) or "Unnamed AI System"

        logger.info(f"Generating action plan for: {system_name}")

        # Build prompt
        user_prompt = self._build_prompt(classification, obligation_map)

        # Call LLM
        raw = self._call_planner(user_prompt)

        # Parse into report
        report = self._parse_report(raw, classification, obligation_map, system_name)

        logger.info(
            f"Action plan: {len(report.actions)} actions, "
            f"{len(report.urgent_actions())} urgent"
        )

        return report

    def _build_prompt(
        self,
        classification: ClassificationResult,
        obligation_map: ObligationMap,
    ) -> str:
        form = classification.intake_form
        system_desc = form.summary() if form else "Unknown system"

        # Format obligations concisely
        ob_lines = []
        for ob in obligation_map.by_deadline():
            ob_lines.append(
                f"  - {ob.article_ref}: {ob.title} "
                f"[{ob.actor}, deadline: {ob.deadline}, "
                f"effort: {ob.theme}]"
            )

        obligations_str = "\n".join(ob_lines)

        return (
            f"AI SYSTEM:\n{system_desc}\n\n"
            f"CLASSIFICATION: {classification.risk_tier.display()} "
            f"(confidence: {classification.confidence:.0%})\n"
            f"Key factors:\n" +
            "\n".join(f"  • {f}" for f in classification.key_factors) +
            f"\n\nAPPLICABLE OBLIGATIONS:\n{obligations_str}\n\n"
            f"Today's date: {TODAY}\n\n"
            f"Generate a prioritised compliance action plan."
        )

    def _call_planner(self, user_prompt: str) -> str:
        response = self._client.chat.complete(
            model=MISTRAL_LARGE_MODEL,
            messages=[
                {"role": "system", "content": ACTION_PLAN_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def _parse_report(
        self,
        raw: str,
        classification: ClassificationResult,
        obligation_map: ObligationMap,
        system_name: str,
    ) -> ComplianceReport:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        form = classification.intake_form
        system_desc = form.raw_description if form else ""

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Action plan parse error: {e}")
            return ComplianceReport(
                system_name=system_name,
                system_description=system_desc,
                classification=classification,
                obligation_map=obligation_map,
                executive_summary="Action plan generation failed — please retry.",
            )

        actions = []
        for i, a in enumerate(data.get("actions", []), 1):
            actions.append(Action(
                priority=a.get("priority", i),
                title=a.get("title", ""),
                description=a.get("description", ""),
                owner=a.get("owner", "provider"),
                deadline=a.get("deadline", "2026-08-02"),
                effort=a.get("effort", "medium"),
                legal_anchor=a.get("legal_anchor", ""),
                notes=a.get("notes", ""),
            ))

        return ComplianceReport(
            system_name=system_name,
            system_description=system_desc,
            classification=classification,
            obligation_map=obligation_map,
            actions=actions,
            executive_summary=data.get("executive_summary", ""),
            key_risks=data.get("key_risks", []),
            immediate_actions=data.get("immediate_actions", []),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_action_planner() -> ActionPlanner:
    return ActionPlanner()
