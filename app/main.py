"""
RAG Evaluation Studio — Streamlit entry point.

This module owns only the app shell: page config, theme toggle,
sidebar navigation, and the landing/overview page. Each feature
(Dataset Builder, Retrieval Evaluation, LLM Evaluation, Experiments,
Reports) lives in its own module under `app/pages/` and is added in
later phases — see `docs/roadmap.md`.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `backend.*` / `app.*` imports when launched via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.components.metric_card import MetricCardData, render_metric_row, section_header
from app.components.styles import inject_theme
from backend.config.settings import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def configure_page() -> None:
    """Set Streamlit page-level configuration. Must run before any other st.* call."""
    settings = get_settings()
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="\U0001F4CA",  # bar chart emoji
        layout="wide",
        initial_sidebar_state="expanded",
    )


def init_session_state() -> None:
    """Set default values for any session_state keys the shell depends on."""
    st.session_state.setdefault("theme_mode", "light")


def render_sidebar() -> str:
    """Render the sidebar: branding, nav, and the theme toggle.

    Returns the label of the currently selected nav item. Real
    navigation wiring (multi-page routing) lands in later phases;
    for now this establishes the shell and information architecture.
    """
    settings = get_settings()

    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1.5rem;">
                <div style="font-size:1.6rem;">📊</div>
                <div>
                    <div style="font-weight:700; font-size:1.05rem; color:var(--rag-text-primary);">
                        {settings.app_name}
                    </div>
                    <div style="font-size:0.75rem; color:var(--rag-text-muted);">
                        v{settings.app_version}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_choice = st.radio(
            "Navigate",
            options=[
                "Overview",
                "Dataset Builder",
                "Retrieval Evaluation",
                "LLM Evaluation",
                "Experiment Manager",
                "Dashboard",
                "Reports",
                "Settings",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        dark_mode = st.toggle("Dark mode", value=(st.session_state["theme_mode"] == "dark"))
        st.session_state["theme_mode"] = "dark" if dark_mode else "light"

        st.divider()
        st.caption("An evaluation platform for RAG pipelines — not another chatbot.")

    return nav_choice


def render_overview() -> None:
    """Landing page: what this tool is, plus placeholder status cards."""
    section_header(
        "RAG Evaluation Studio",
        "Evaluate retrieval quality, faithfulness, and cost across RAG configurations — side by side.",
    )

    render_metric_row(
        [
            MetricCardData(label="Datasets", value="0", tone="info", help_text="Evaluation datasets built"),
            MetricCardData(label="Experiments", value="0", tone="info", help_text="Configurations run"),
            MetricCardData(label="Avg. Faithfulness", value="—", tone="info", help_text="Across latest run"),
            MetricCardData(label="Avg. Latency", value="—", tone="info", help_text="Per query, latest run"),
        ]
    )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        st.markdown(
            """
            <div class="rag-card">
                <h4 style="margin-top:0;">What this tool does</h4>
                <p style="color: var(--rag-text-secondary); line-height:1.6;">
                    RAG Evaluation Studio scores your Retrieval-Augmented Generation pipeline
                    on retrieval quality (Precision@K, Recall@K, MRR, nDCG), generation quality
                    (Faithfulness, Answer Relevancy, Context Precision/Recall, Hallucination),
                    and operational cost (latency, tokens, $). It then lets you compare multiple
                    configurations — chunk size, embedding model, vector store, Top-K, reranker,
                    LLM — to find the setup that performs best on <em>your</em> data.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="rag-card">
                <h4 style="margin-top:0;">Build status</h4>
                <p style="color: var(--rag-text-secondary); font-size: 0.9rem; line-height:1.8;">
                    ✅ Phase 1 — Project structure &amp; setup<br>
                    ⬜ Phase 2 — Dataset Builder<br>
                    ⬜ Phase 3 — Retrieval Metrics<br>
                    ⬜ Phase 4 — LLM Evaluation<br>
                    ⬜ Phase 5 — Dashboard<br>
                    ⬜ Phase 6 — Experiment Comparison<br>
                    ⬜ Phase 7 — Reporting<br>
                    ⬜ Phase 8 — Testing &amp; Polish
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_placeholder(page_name: str) -> None:
    """Placeholder for pages not yet built in the current phase."""
    section_header(page_name, "This section is built in a later phase — see the roadmap in the README.")
    st.info(f"**{page_name}** is coming in an upcoming phase. The navigation and shell are already wired up.")


def main() -> None:
    configure_page()
    init_session_state()
    inject_theme()

    nav_choice = render_sidebar()
    logger.debug(f"Rendering page: {nav_choice}")

    if nav_choice == "Overview":
        render_overview()
    else:
        render_placeholder(nav_choice)


if __name__ == "__main__":
    main()
