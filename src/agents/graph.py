"""
graph.py — LangGraph agentic pipeline for AI Act Navigator

Wires the four agents into a conditional state graph:

  free_text_input
        ↓
  [EXTRACT]  — AISystemExtractor → IntakeForm + confidence scores
        ↓
  [GATE]     — confidence check
        ├── low confidence → [CLARIFY] → back to GATE
        └── sufficient     → continue
        ↓
  [CLASSIFY] — RiskClassifier → ClassificationResult
        ↓
  [MAP]      — ObligationMapper → ObligationMap
        ↓
  [PLAN]     — ActionPlanner → ComplianceReport
        ↓
  ComplianceReport (final output)

The CLARIFY node implements the confidence-gated loop (max 2 rounds).
Each node is a pure function: state_in → state_out.
"""

import logging
import os
from typing import TypedDict, Optional, Annotated
from dataclasses import dataclass

from dotenv import load_dotenv

from .extractor import AISystemExtractor, IntakeForm
from .classifier import RiskClassifier, ClassificationResult, get_classifier
from .obligation_mapper import ObligationMapper, ObligationMap, get_obligation_mapper
from .action_planner import ActionPlanner, ComplianceReport, get_action_planner

load_dotenv()
logger = logging.getLogger(__name__)

MAX_CLARIFICATION_ROUNDS = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "2"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """
    State object passed between pipeline nodes.
    Each node reads what it needs and adds its output.
    """
    # Input
    raw_description: str = ""
    clarification_responses: list[str] = None

    # Intermediate outputs
    intake_form: Optional[IntakeForm] = None
    classification: Optional[ClassificationResult] = None
    obligation_map: Optional[ObligationMap] = None

    # Final output
    report: Optional[ComplianceReport] = None

    # Pipeline control
    clarification_round: int = 0
    clarification_question: Optional[str] = None
    clarification_options: list[str] = None
    needs_clarification: bool = False
    error: Optional[str] = None

    def __post_init__(self):
        if self.clarification_responses is None:
            self.clarification_responses = []
        if self.clarification_options is None:
            self.clarification_options = []


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------

def extract_node(state: PipelineState, extractor: AISystemExtractor) -> PipelineState:
    """
    Extract structured intake form from description.
    Incorporates any clarification responses from previous rounds.
    """
    # Build full context including any clarifications
    full_description = state.raw_description
    if state.clarification_responses:
        context_lines = [f"Original description: {state.raw_description}"]
        context_lines.append("Additional context:")
        context_lines.extend(state.clarification_responses)
        full_description = "\n".join(context_lines)

    try:
        state.intake_form = extractor.extract(full_description)
        logger.info(
            f"Extraction complete — "
            f"ready: {state.intake_form.is_ready_for_classification()}"
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        state.error = f"Extraction failed: {e}"

    return state


def confidence_gate_node(state: PipelineState) -> PipelineState:
    """
    Check if intake form has sufficient confidence to proceed.
    Sets needs_clarification flag and generates the question if needed.
    """
    if not state.intake_form:
        state.error = "No intake form available"
        return state

    if state.clarification_round >= MAX_CLARIFICATION_ROUNDS:
        # Max rounds reached — proceed with what we have
        logger.info(
            f"Max clarification rounds reached ({MAX_CLARIFICATION_ROUNDS}). "
            f"Proceeding with uncertain fields: "
            f"{state.intake_form.get_uncertain_critical_fields()}"
        )
        state.needs_clarification = False
        return state

    if state.intake_form.is_ready_for_classification():
        state.needs_clarification = False
        return state

    # Generate clarification question for most uncertain field
    uncertain = state.intake_form.get_uncertain_critical_fields()
    if not uncertain:
        state.needs_clarification = False
        return state

    target_field = min(
        uncertain,
        key=lambda f: state.intake_form.confidence.get(f, 0.0)
    )
    confidence = state.intake_form.confidence.get(target_field, 0.0)

    # Generate targeted question based on field
    question, options = _generate_clarification_question(
        target_field,
        getattr(state.intake_form, target_field),
        confidence,
    )

    state.needs_clarification = True
    state.clarification_question = question
    state.clarification_options = options
    logger.info(
        f"Clarification needed: {target_field} "
        f"(confidence: {confidence:.0%})"
    )

    return state


def classify_node(
    state: PipelineState, classifier: RiskClassifier
) -> PipelineState:
    """Classify the AI system's risk tier."""
    if not state.intake_form or state.error:
        return state

    try:
        state.classification = classifier.classify(state.intake_form)
        logger.info(
            f"Classification: {state.classification.risk_tier.display()} "
            f"({state.classification.confidence:.0%})"
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        state.error = f"Classification failed: {e}"

    return state


def map_obligations_node(
    state: PipelineState, mapper: ObligationMapper
) -> PipelineState:
    """Map applicable obligations for the classified system."""
    if not state.classification or state.error:
        return state

    try:
        state.obligation_map = mapper.map_obligations(state.classification)
        logger.info(
            f"Obligation mapping: "
            f"{state.obligation_map.mandatory_count()} mandatory obligations"
        )
    except Exception as e:
        logger.error(f"Obligation mapping failed: {e}")
        state.error = f"Obligation mapping failed: {e}"

    return state


def plan_actions_node(
    state: PipelineState, planner: ActionPlanner
) -> PipelineState:
    """Generate the compliance action plan."""
    if not state.obligation_map or state.error:
        return state

    try:
        state.report = planner.plan(state.classification, state.obligation_map)
        logger.info(
            f"Action plan: {state.report.action_count()} actions, "
            f"{len(state.report.urgent_actions())} urgent"
        )
    except Exception as e:
        logger.error(f"Action planning failed: {e}")
        state.error = f"Action planning failed: {e}"

    return state


# ---------------------------------------------------------------------------
# Clarification question templates
# ---------------------------------------------------------------------------

def _generate_clarification_question(
    field: str,
    current_value: Optional[str],
    confidence: float,
) -> tuple[str, list[str]]:
    """
    Generate a targeted clarification question for an uncertain field.
    Returns (question_text, answer_options).
    """
    templates = {
        "sector": (
            "What is the primary sector or application domain of your AI system?",
            [
                "Education / vocational training",
                "Healthcare / mental health support",
                "Employment / HR management",
                "Essential public/private services",
                "Critical infrastructure",
                "Other",
            ],
        ),
        "user_base": (
            "Who are the primary users of your AI system?",
            [
                "General public (no specific vulnerability)",
                "Minors (under 18)",
                "Vulnerable groups (disabilities, mental health, etc.)",
                "Professionals / trained operators",
                "Employees",
                "Mixed audience",
            ],
        ),
        "autonomy_level": (
            "How does your AI system make or support decisions?",
            [
                "Autonomous — makes decisions independently without human review",
                "Advisory — provides recommendations, human makes final decision",
                "Informational — provides information only, no decision support",
            ],
        ),
        "deployment_status": (
            "What is the current deployment status of your system?",
            [
                "Deployed publicly (accessible by anyone)",
                "Deployed in restricted environment (specific users/org)",
                "Internal use only (within your organisation)",
                "Not yet deployed (development/prototype phase)",
            ],
        ),
    }

    question, options = templates.get(
        field,
        (
            f"Could you clarify the '{field}' aspect of your AI system?",
            [],
        ),
    )

    if current_value:
        question += f"\n(Current best assessment: {current_value!r}, confidence: {confidence:.0%})"

    return question, options


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

class NavigatorPipeline:
    """
    The complete AI Act Navigator pipeline.

    Provides two modes:
    1. run_silent() — batch mode, no clarification, proceeds with best data
    2. run_interactive() — interactive mode, pauses for clarification input
    """

    def __init__(
        self,
        extractor: AISystemExtractor,
        classifier: RiskClassifier,
        mapper: ObligationMapper,
        planner: ActionPlanner,
    ):
        self._extractor = extractor
        self._classifier = classifier
        self._mapper = mapper
        self._planner = planner

    @classmethod
    def from_env(cls, collection_name: str = "ai_act_navigator") -> "NavigatorPipeline":
        """Build pipeline from environment variables."""
        return cls(
            extractor=AISystemExtractor(),
            classifier=get_classifier(collection_name),
            mapper=get_obligation_mapper(collection_name),
            planner=get_action_planner(),
        )

    def run_silent(self, description: str) -> ComplianceReport:
        """
        Run full pipeline without clarification.
        Proceeds with whatever confidence level is achieved.
        Used for batch processing and testing.
        """
        state = PipelineState(raw_description=description)
        state = extract_node(state, self._extractor)
        state = confidence_gate_node(state)
        # Skip clarification — proceed regardless
        state = classify_node(state, self._classifier)
        state = map_obligations_node(state, self._mapper)
        state = plan_actions_node(state, self._planner)

        if state.error:
            raise RuntimeError(state.error)

        return state.report

    def run_interactive(
        self,
        description: str,
        clarification_callback,
    ) -> ComplianceReport:
        """
        Run pipeline with interactive clarification loop.

        Args:
            description:             initial system description
            clarification_callback:  fn(question: str, options: list) → str
                                     receives question, returns user answer

        Returns:
            ComplianceReport
        """
        state = PipelineState(raw_description=description)

        # Extract → gate → clarify loop
        for round_num in range(MAX_CLARIFICATION_ROUNDS + 1):
            state = extract_node(state, self._extractor)
            state = confidence_gate_node(state)

            if not state.needs_clarification:
                break

            # Ask for clarification
            answer = clarification_callback(
                state.clarification_question,
                state.clarification_options,
            )
            state.clarification_responses.append(
                f"On {state.clarification_question}: {answer}"
            )
            state.clarification_round = round_num + 1

        # Classify → map → plan
        state = classify_node(state, self._classifier)
        state = map_obligations_node(state, self._mapper)
        state = plan_actions_node(state, self._planner)

        if state.error:
            raise RuntimeError(state.error)

        return state.report


# ---------------------------------------------------------------------------
# CLI — end-to-end test with Complice
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    complice_description = (
        "Complice is a RAG-based conversational assistant for young adults "
        "with ASD aged 16-25, not deployed on a public web platform, built "
        "on OpenAI, offering emotional support and information on mental "
        "health topics."
    )

    print("="*65)
    print("AI Act Navigator — Full Pipeline Test")
    print("="*65)
    print(f"\nInput: {complice_description}\n")

    pipeline = NavigatorPipeline.from_env()

    print("Running pipeline (silent mode)...\n")
    report = pipeline.run_silent(complice_description)

    print(report.full_summary())

    import json
    print("\n" + "="*65)
    print("JSON output (for UI):")
    print("="*65)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False)[:2000])
    print("... (truncated)")
