"""
app.py — Main Streamlit application for AI Act Navigator

4-step compliance assessment flow:
  Step 1 — INPUT:     Free-text description + example chips
  Step 2 — REVIEW:    Editable intake form with confidence bars
  Step 3 — CLASSIFY:  Risk tier determination with legal reasoning
  Step 4 — REPORT:    Full compliance report + action plan + download

Session state keys:
  step          : int (1-4)
  description   : str
  form          : IntakeForm
  classification: ClassificationResult
  obligation_map: ObligationMap
  report        : ComplianceReport
  error         : str or None
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Add project root to sys.path for module imports
sys.path.insert(0, str(Path(__file__).parents[3]))

load_dotenv()

from src.ui.components import (
    inject_css,
    render_header,
    render_steps,
    render_input_step,
    render_form_review,
    render_classification,
    render_report,
    render_error,
)
from src.agents.extractor import AISystemExtractor
from src.agents.classifier import get_classifier
from src.agents.obligation_mapper import get_obligation_mapper
from src.agents.action_planner import get_action_planner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Act Navigator",
    page_icon="🇪🇺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Cached resource initialisation
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI Act Navigator...")
def load_pipeline():
    """
    Load all pipeline components once and cache them.
    st.cache_resource persists across user sessions — the cross-encoder
    model and Qdrant client are initialised only once per server restart.
    """
    return {
        "extractor":  AISystemExtractor(),
        "classifier": get_classifier(),
        "mapper":     get_obligation_mapper(),
        "planner":    get_action_planner(),
    }

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_state():
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "error" not in st.session_state:
        st.session_state.error = None


def go_to(step: int):
    st.session_state.step = step
    st.session_state.error = None

# ---------------------------------------------------------------------------
# Pipeline runners — each wrapped in try/except with user-friendly errors
# ---------------------------------------------------------------------------

def run_extraction(description: str, pipeline: dict):
    """Extract IntakeForm from description."""
    with st.spinner("🔍 Extracting system attributes..."):
        try:
            form = pipeline["extractor"].extract(description)
            st.session_state.form = form
            st.session_state.description = description
            go_to(2)
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            st.session_state.error = (
                f"Extraction failed: {e}. "
                "Please check your Mistral API key."
            )


def run_classification(pipeline: dict):
    """Classify risk tier from confirmed IntakeForm."""
    with st.spinner("⚖️ Classifying risk tier..."):
        try:
            result = pipeline["classifier"].classify(st.session_state.form)
            st.session_state.classification = result
            go_to(3)
        except Exception as e:
            logger.error(f"Classification error: {e}")
            st.session_state.error = f"Classification failed: {e}"


def run_report(pipeline: dict):
    """Generate full compliance report."""
    with st.spinner("📋 Generating compliance report..."):
        try:
            ob_map = pipeline["mapper"].map_obligations(
                st.session_state.classification
            )
            st.session_state.obligation_map = ob_map

            report = pipeline["planner"].plan(
                st.session_state.classification,
                ob_map,
            )
            st.session_state.report = report
            go_to(4)
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            st.session_state.error = f"Report generation failed: {e}"

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    init_state()
    inject_css()
    render_header()

    # Load pipeline (cached)
    try:
        pipeline = load_pipeline()
    except Exception as e:
        render_error(
            f"Failed to initialise pipeline: {e}. "
            "Check MISTRAL_API_KEY, QDRANT_URL, and QDRANT_API_KEY in your .env file."
        )
        return

    step = st.session_state.step
    render_steps(step)

    # Show persistent error if any
    if st.session_state.error:
        render_error(st.session_state.error)
        st.session_state.error = None

    # -----------------------------------------------------------------------
    # STEP 1 — Input
    # -----------------------------------------------------------------------
    if step == 1:
        description = render_input_step()
        if description:
            run_extraction(description, pipeline)
            st.rerun()

    # -----------------------------------------------------------------------
    # STEP 2 — Review intake form
    # -----------------------------------------------------------------------
    elif step == 2:
        if "form" not in st.session_state:
            go_to(1)
            st.rerun()

        form, confirmed = render_form_review(st.session_state.form)
        st.session_state.form = form

        if confirmed:
            run_classification(pipeline)
            st.rerun()

    # -----------------------------------------------------------------------
    # STEP 3 — Classification result
    # -----------------------------------------------------------------------
    elif step == 3:
        if "classification" not in st.session_state:
            go_to(1)
            st.rerun()

        render_classification(st.session_state.classification)

        st.markdown("")
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button(
                "📋 Generate full compliance report",
                type="primary",
                use_container_width=True,
            ):
                run_report(pipeline)
                st.rerun()
        with c2:
            if st.button("← Back", use_container_width=True):
                go_to(2)
                st.rerun()

    # -----------------------------------------------------------------------
    # STEP 4 — Full report
    # -----------------------------------------------------------------------
    elif step == 4:
        if "report" not in st.session_state:
            go_to(1)
            st.rerun()

        render_report(st.session_state.report)

        st.markdown("")
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button(
                "🔄 Assess another system",
                use_container_width=True,
            ):
                st.session_state.clear()
                st.rerun()
        with c2:
            if st.button("← Back", use_container_width=True):
                go_to(3)
                st.rerun()

    # -----------------------------------------------------------------------
    # Sidebar — about + links
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        **AI Act Navigator** assesses AI systems against
        [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng)
        using a RAG pipeline grounded in the official legal text.

        **How it works:**
        1. Extracts system attributes from your description
        2. Retrieves relevant AI Act provisions
        3. Classifies risk tier with legal reasoning
        4. Maps applicable obligations and deadlines
        5. Generates a prioritised compliance action plan

        **Tech stack:**
        - Mistral AI (extraction + reasoning)
        - Qdrant (hybrid vector search)
        - LangGraph (agentic pipeline)
        - RAGAS (evaluation)
        """)

        st.divider()
        st.markdown("### Key deadlines")
        st.markdown("""
        | Deadline | What applies |
        |----------|-------------|
        | ✅ Feb 2025 | Prohibited practices |
        | ✅ Aug 2025 | GPAI obligations |
        | ⏳ **Aug 2026** | High-risk systems |
        | ⏳ Aug 2027 | Products (Annex II) |
        """)

        st.divider()
        st.caption(
            "Built by [Agnès Heijligers](https://linkedin.com/in/agaheijligers) · "
            "[GitHub](https://github.com/AgaHei/ai-act-navigator)"
        )


if __name__ == "__main__":
    main()
