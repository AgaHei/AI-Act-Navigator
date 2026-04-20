"""
components.py — Reusable UI components for AI Act Navigator

All visual building blocks used by app.py:
  - Risk tier badge
  - Confidence bars
  - Intake form editor
  - Classification result card
  - Obligation list
  - Action plan checklist
  - Report download buttons
"""

import json
import streamlit as st
from datetime import date
from typing import Optional

from ..agents.extractor import IntakeForm
from ..agents.classifier import ClassificationResult, RiskTier
from ..agents.obligation_mapper import ObligationMap
from ..agents.action_planner import ComplianceReport, Action

TODAY = date.today().isoformat()

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

TIER_STYLES = {
    RiskTier.PROHIBITED:   ("#7f1d1d", "#fca5a5", "🚫"),
    RiskTier.HIGH_RISK:    ("#7c2d12", "#fdba74", "🔴"),
    RiskTier.LIMITED_RISK: ("#713f12", "#fde68a", "🟡"),
    RiskTier.GPAI:         ("#4c1d95", "#c4b5fd", "🟣"),
    RiskTier.MINIMAL_RISK: ("#14532d", "#86efac", "🟢"),
    RiskTier.UNCERTAIN:    ("#1e3a5f", "#93c5fd", "❓"),
}

EFFORT_COLORS = {
    "low": "#bbf7d0", "medium": "#fef08a", "high": "#fecaca"
}

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

    :root {
        --eu-blue: #003399;
        --eu-gold: #FFCC00;
        --surface: #f8f9fc;
        --border: #e2e8f0;
        --radius: 10px;
        --shadow: 0 2px 12px rgba(0,51,153,0.07);
        --font-display: 'DM Serif Display', serif;
        --font-body: 'DM Sans', sans-serif;
    }

    .main .block-container { max-width: 860px; padding-top: 1.5rem; }
    
    /* Enhanced readability - larger fonts */
    .main .block-container p, .main .block-container div, .main .block-container li {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        font-size: 15px !important;
    }
    
    .stButton button {
        font-size: 15px !important;
        font-weight: 500 !important;
    }

    .nav-header {
        padding: 1rem 0 0.75rem;
        border-bottom: 3px solid var(--eu-gold);
        margin-bottom: 1.5rem;
    }
    .nav-title {
        font-family: var(--font-display);
        font-size: 2rem;
        color: var(--eu-blue);
        margin: 0;
        line-height: 1.2;
    }
    .nav-sub {
        font-family: var(--font-body);
        font-size: 0.88rem;
        color: #64748b;
        margin: 4px 0 0;
    }

    .step-bar {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 1.75rem;
    }
    .step {
        display: flex;
        align-items: center;
        gap: 6px;
        font-family: var(--font-body);
        font-size: 0.8rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .step.active { color: var(--eu-blue); font-weight: 600; }
    .step.done   { color: #16a34a; }
    .step-num {
        width: 22px; height: 22px; border-radius: 50%;
        background: #e2e8f0; display: flex;
        align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700;
    }
    .step-num.active { background: var(--eu-blue); color: white; }
    .step-num.done   { background: #16a34a; color: white; }
    .sep { flex: 1; height: 1px; background: #e2e8f0; min-width: 20px; }

    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: var(--radius);
        padding: 10px 14px;
        font-family: var(--font-body);
        font-size: 0.84rem;
        color: #1e40af;
        margin-bottom: 1rem;
    }

    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 14px 22px;
        border-radius: var(--radius);
        border: 2px solid;
        margin: 0.5rem 0 1.25rem;
    }
    .risk-badge-label {
        font-family: var(--font-display);
        font-size: 1.35rem;
        line-height: 1.2;
    }
    .risk-badge-meta {
        font-family: var(--font-body);
        font-size: 0.78rem;
        opacity: 0.75;
        margin-top: 2px;
    }

    .conf-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 5px 0;
        font-family: var(--font-body);
        font-size: 0.82rem;
    }
    .conf-label { width: 155px; color: #475569; text-transform: capitalize; }
    .conf-track {
        flex: 1; height: 5px;
        background: #e2e8f0; border-radius: 3px; overflow: hidden;
    }
    .conf-fill { height: 100%; border-radius: 3px; }
    .conf-val { width: 36px; text-align: right; font-weight: 600; color: #334155; }

    .ob-item {
        border-left: 3px solid var(--eu-blue);
        padding: 9px 13px;
        margin: 7px 0;
        background: var(--surface);
        border-radius: 0 var(--radius) var(--radius) 0;
    }
    .ob-ref {
        font-family: var(--font-body); font-size: 0.72rem;
        font-weight: 600; color: var(--eu-blue);
        text-transform: uppercase; letter-spacing: 0.4px;
    }
    .ob-title {
        font-family: var(--font-body); font-size: 0.92rem;
        font-weight: 600; color: #1e293b; margin: 2px 0;
    }
    .ob-desc { font-family: var(--font-body); font-size: 0.83rem; color: #475569; }
    .ob-meta { font-family: var(--font-body); font-size: 0.72rem; color: #94a3b8; margin-top: 3px; }

    .act-item {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 13px 15px;
        margin: 7px 0;
        background: white;
    }
    .act-head { display: flex; align-items: center; gap: 9px; margin-bottom: 5px; }
    .act-num {
        width: 24px; height: 24px; border-radius: 50%;
        background: var(--eu-blue); color: white;
        display: flex; align-items: center; justify-content: center;
        font-family: var(--font-body); font-size: 0.72rem; font-weight: 700;
        flex-shrink: 0;
    }
    .act-title { font-family: var(--font-body); font-size: 0.92rem; font-weight: 600; color: #1e293b; }
    .act-desc { font-family: var(--font-body); font-size: 0.83rem; color: #475569; padding-left: 33px; margin: 3px 0; }
    .act-tags { display: flex; gap: 8px; flex-wrap: wrap; padding-left: 33px; margin-top: 5px; }
    .tag {
        font-family: var(--font-body); font-size: 0.7rem; font-weight: 600;
        padding: 2px 7px; border-radius: 4px;
        background: #f1f5f9; color: #475569;
    }
    .tag.overdue { background: #fee2e2; color: #991b1b; }
    .tag.urgent  { background: #fef3c7; color: #92400e; }
    .tag.ok      { background: #dcfce7; color: #14532d; }

    .sec-head {
        font-family: var(--font-display);
        font-size: 1.25rem; color: #1e293b;
        padding-bottom: 7px;
        border-bottom: 2px solid var(--eu-gold);
        margin: 1.5rem 0 0.75rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header + steps
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="nav-header">
        <div style="display:flex;align-items:center;gap:14px">
            <span style="font-size:2.4rem">🇪🇺</span>
            <div>
                <h1 class="nav-title">AI Act Navigator</h1>
                <p class="nav-sub">EU AI Act compliance assessment · Powered by RAG + LangGraph agents</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_steps(current: int):
    steps = [(1,"Describe"),(2,"Review"),(3,"Classify"),(4,"Report")]
    html = '<div class="step-bar">'
    for i,(n,label) in enumerate(steps):
        cls = "done" if n < current else ("active" if n == current else "")
        icon = "✓" if n < current else str(n)
        html += f'<div class="step {cls}"><div class="step-num {cls}">{icon}</div>{label}</div>'
        if i < len(steps)-1:
            html += '<div class="sep"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Step 1 — Input
# ---------------------------------------------------------------------------

def render_input_step() -> Optional[str]:
    st.markdown('<div class="sec-head">Describe your AI system</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    📝 Describe your AI system in plain language — what it does, who uses it,
    how it's deployed, and what technology powers it.
    </div>""", unsafe_allow_html=True)

    # Example chips
    st.markdown("**Try an example:**")
    examples = [
        "Complice — RAG assistant for young adults with ASD, mental health support, built on OpenAI, not yet deployed publicly",
        "HR screening tool that ranks job applicants using CV analysis, deployed internally for a large corporation",
        "Customer service chatbot for an e-commerce platform, built on GPT-4, deployed publicly",
        "Medical imaging AI that assists radiologists in detecting tumour anomalies, hospital deployment",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"📎 {ex[:52]}…", key=f"ex_{i}", use_container_width=True):
                st.session_state.description = ex
                st.rerun()

    description = st.text_area(
        "description",
        value=st.session_state.get("description", ""),
        height=130,
        placeholder="e.g. 'A conversational AI assistant that helps HR professionals screen "
                    "job applications, deployed internally, built using GPT-4 via Azure OpenAI...'",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([3,1])
    with col1:
        go = st.button("🔍 Analyse system", type="primary",
                       use_container_width=True,
                       disabled=len(description.strip()) < 20)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if go and description.strip():
        return description.strip()
    if description.strip() and len(description.strip()) < 20:
        st.caption("Please add a bit more detail (at least 20 characters)")
    return None


# ---------------------------------------------------------------------------
# Step 2 — Form review
# ---------------------------------------------------------------------------

def render_form_review(form: IntakeForm):
    st.markdown('<div class="sec-head">Review extracted information</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    ✅ Review and correct any fields before classification runs.
    The confidence bars show how certain the extraction was.
    </div>""", unsafe_allow_html=True)

    _render_confidence_bars(form.confidence)
    st.markdown("---")
    st.markdown("**Edit if needed:**")

    c1, c2 = st.columns(2)
    opts = {
        "sector": ["education","healthcare","employment","critical_infrastructure",
                   "law_enforcement","migration","justice","biometric","other"],
        "user_base": ["general_public","vulnerable_groups","minors","professionals",
                      "employees","students","mixed"],
        "autonomy": ["autonomous","advisory","informational"],
        "deployment": ["deployed_public","deployed_restricted","internal_use",
                       "not_deployed","prototype"],
        "actor": ["provider","deployer","both","unknown"],
        "eu": ["yes","no","unknown"],
    }

    with c1:
        form.system_name = st.text_input("System name", value=form.system_name or "")
        form.sector = st.selectbox("Primary sector", opts["sector"],
                                   index=_idx(opts["sector"], form.sector or "other"))
        form.autonomy_level = st.selectbox("Autonomy level", opts["autonomy"],
                                           index=_idx(opts["autonomy"], form.autonomy_level or "advisory"),
                                           help="autonomous=decides alone | advisory=recommends | informational=info only")
        form.actor_role = st.selectbox("Your role", opts["actor"],
                                       index=_idx(opts["actor"], form.actor_role or "provider"))

    with c2:
        form.user_base = st.selectbox("User base", opts["user_base"],
                                      index=_idx(opts["user_base"], form.user_base or "general_public"))
        form.deployment_status = st.selectbox("Deployment status", opts["deployment"],
                                              index=_idx(opts["deployment"], form.deployment_status or "not_deployed"))
        form.gpai_model = st.text_input("GPAI model used", value=form.gpai_model or "",
                                        placeholder="e.g. OpenAI GPT-4, Mistral Large…") or None
        form.targets_eu_market = st.selectbox("Targets EU market?", opts["eu"],
                                              index=_idx(opts["eu"], form.targets_eu_market or "unknown"))

    data_opts = ["text","biometric","health","financial","location","behavioral","other"]
    form.data_types = st.multiselect("Data types processed", data_opts,
                                     default=[d for d in (form.data_types or []) if d in data_opts])

    st.markdown("")
    c1, c2 = st.columns([3,1])
    with c1:
        confirmed = st.button("✅ Confirm and classify", type="primary", use_container_width=True)
    with c2:
        back = st.button("← Back", use_container_width=True)

    if back:
        for k in ["form","step"]:
            st.session_state.pop(k, None)
        st.rerun()

    return form, confirmed


def _render_confidence_bars(conf: dict):
    primary_fields = ["sector","user_base","autonomy_level","deployment_status"]
    html = '<div style="margin-bottom:12px">'
    for f in primary_fields:
        score = conf.get(f, 0.0)
        pct = int(score * 100)
        color = "#22c55e" if score >= 0.8 else ("#f59e0b" if score >= 0.6 else "#ef4444")
        flag = "✅" if score >= 0.8 else ("⚠️" if score >= 0.6 else "❌")
        html += f"""<div class="conf-row">
            <span class="conf-label">{f.replace('_',' ')}</span>
            <div class="conf-track"><div class="conf-fill" style="width:{pct}%;background:{color}"></div></div>
            <span class="conf-val">{pct}%</span>
            <span>{flag}</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Step 3 — Classification
# ---------------------------------------------------------------------------

def render_classification(result: ClassificationResult):
    st.markdown('<div class="sec-head">Risk classification</div>', unsafe_allow_html=True)

    tier = result.risk_tier
    dark, light, icon = TIER_STYLES.get(tier, ("#1e3a5f","#93c5fd","❓"))

    st.markdown(f"""
    <div class="risk-badge" style="background:{dark}18;border-color:{dark};color:{dark}">
        <span style="font-size:1.8rem">{icon}</span>
        <div>
            <div class="risk-badge-label">{tier.display()}</div>
            <div class="risk-badge-meta">Confidence: {result.confidence:.0%}
            &nbsp;·&nbsp; Deadline: {tier.deadline()}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if result.key_factors:
        st.markdown("**Key factors:**")
        for f in result.key_factors:
            st.markdown(f"• {f}")

    if result.primary_articles:
        st.markdown(f"**Legal basis:** `{'  ·  '.join(result.primary_articles)}`")

    if result.annex_iii_domain:
        st.markdown(f"**Annex III:** {result.annex_iii_domain} (point {result.annex_iii_point})")

    if result.transparency_obligations:
        st.info(f"📋 **Art. 50 transparency obligations apply** — "
                f"{result.transparency_reasoning or 'AI interaction disclosure required'}")

    if result.gpai_dependency_noted:
        st.info(f"🔗 **GPAI model dependency noted** — "
                f"{result.gpai_dependency_reasoning or 'Document foundation model usage'}")

    if result.borderline_considerations:
        with st.expander("⚠️ Borderline considerations"):
            for note in result.borderline_considerations:
                st.markdown(f"• {note}")

    with st.expander("📖 Full legal reasoning"):
        st.markdown(result.reasoning)

    if result.retrieved_chunks:
        with st.expander(f"📚 Legal sources consulted ({len(result.retrieved_chunks)} provisions)"):
            for chunk in result.retrieved_chunks[:6]:
                st.markdown(f"**{chunk.display_reference}** (score: {chunk.score:.3f})")
                st.caption(chunk.text[:200] + "…")
                st.divider()


# ---------------------------------------------------------------------------
# Step 4 — Full report
# ---------------------------------------------------------------------------

def render_report(report: ComplianceReport):
    st.markdown('<div class="sec-head">Compliance assessment report</div>', unsafe_allow_html=True)

    # Executive summary
    with st.container(border=True):
        st.markdown("**Executive summary**")
        st.markdown(report.executive_summary)

    if report.key_risks:
        with st.expander("⚠️ Key risks if non-compliant"):
            for r in report.key_risks:
                st.markdown(f"• {r}")

    # Obligations
    if report.obligation_map and report.obligation_map.obligations:
        st.markdown('<div class="sec-head">Applicable obligations</div>', unsafe_allow_html=True)
        _render_obligations(report.obligation_map)

    # Actions
    st.markdown('<div class="sec-head">Compliance action plan</div>', unsafe_allow_html=True)
    urgent_n = len(report.urgent_actions())
    st.markdown(f"**{report.action_count()} actions** · "
                f"**{urgent_n} urgent** (deadline ≤ Aug 2026)")
    _render_actions(report.actions)

    st.divider()
    _render_download(report)


def _render_obligations(om: ObligationMap):
    for theme, obs in om.by_theme().items():
        mandatory_n = sum(1 for o in obs if o.is_mandatory)
        with st.expander(
            f"**{theme.replace('_',' ').title()}** — "
            f"{mandatory_n} mandatory, {len(obs)-mandatory_n} recommended",
            expanded=mandatory_n > 0,
        ):
            for ob in obs:
                flag = "✅" if ob.is_mandatory else "○"
                annex = f" + {', '.join(ob.annex_refs)}" if ob.annex_refs else ""
                st.markdown(f"""
                <div class="ob-item">
                    <div class="ob-ref">{ob.article_ref}{annex}</div>
                    <div class="ob-title">{flag} {ob.title}</div>
                    <div class="ob-desc">{ob.description}</div>
                    <div class="ob-meta">[{ob.actor}] · Deadline: {ob.deadline}</div>
                </div>""", unsafe_allow_html=True)


def _render_actions(actions: list):
    for action in sorted(actions, key=lambda a: (a.deadline, a.priority)):
        urgency = action.urgency_label
        tag_cls = ("overdue" if "OVERDUE" in urgency
                   else "urgent" if "URGENT" in urgency or "HIGH" in urgency
                   else "ok")
        effort_bg = EFFORT_COLORS.get(action.effort, "#f1f5f9")
        st.markdown(f"""
        <div class="act-item">
            <div class="act-head">
                <div class="act-num">{action.priority}</div>
                <div class="act-title">{action.title}</div>
            </div>
            <div class="act-desc">{action.description}</div>
            <div class="act-tags">
                <span class="tag {tag_cls}">{urgency}</span>
                <span class="tag">📅 {action.deadline}</span>
                <span class="tag">👤 {action.owner}</span>
                <span class="tag" style="background:{effort_bg}40">effort: {action.effort}</span>
                <span class="tag">⚖️ {action.legal_anchor}</span>
            </div>
        </div>""", unsafe_allow_html=True)
        if action.notes:
            st.caption(f"💡 {action.notes}")


def _render_download(report: ComplianceReport):
    st.markdown("**Download report:**")
    c1, c2 = st.columns(2)
    safe_name = (report.system_name or "report").lower().replace(" ", "_")
    with c1:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            file_name=f"ai_act_{safe_name}.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "⬇️ Download Markdown",
            data=_to_markdown(report),
            file_name=f"ai_act_{safe_name}.md",
            mime="text/markdown",
            use_container_width=True,
        )


def _to_markdown(report: ComplianceReport) -> str:
    lines = [
        f"# AI Act Compliance Assessment — {report.system_name}",
        f"*Generated by AI Act Navigator · {report.assessment_date}*",
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
        f"## Risk Classification",
        f"**Tier:** {report.risk_tier().display()}",
    ]
    if report.classification:
        lines += [
            f"**Confidence:** {report.classification.confidence:.0%}",
            f"**Legal basis:** {', '.join(report.classification.primary_articles)}",
            "",
            "### Legal Reasoning",
            report.classification.reasoning,
            "",
            "### Key Factors",
        ] + [f"- {f}" for f in report.classification.key_factors]

    lines += ["", "## Applicable Obligations"]
    if report.obligation_map:
        for ob in report.obligation_map.by_deadline():
            flag = "✅" if ob.is_mandatory else "○"
            lines.append(f"- {flag} **{ob.article_ref}** — {ob.title} [{ob.actor}] deadline: {ob.deadline}")

    lines += ["", "## Compliance Action Plan"]
    for a in sorted(report.actions, key=lambda x: (x.deadline, x.priority)):
        lines += [
            f"### {a.priority}. {a.title}",
            a.description,
            f"- **Owner:** {a.owner}",
            f"- **Deadline:** {a.deadline}",
            f"- **Effort:** {a.effort}",
            f"- **Legal anchor:** {a.legal_anchor}",
            "",
        ]
    lines += [
        "---",
        "*Generated by AI Act Navigator*",
        "*https://github.com/AgaHei/ai-act-navigator*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _idx(options: list, value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def render_error(msg: str):
    st.error(f"❌ {msg}")
    st.markdown("""<div class="info-box">
    💡 Try refreshing or rephrasing your description.
    If the error persists, check your API key configuration.
    </div>""", unsafe_allow_html=True)
