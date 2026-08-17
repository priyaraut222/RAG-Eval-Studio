"""
Reports page.

Pick one saved experiment run for a full single-run report (CSV,
JSON, Markdown, PDF), or several for a comparison report (CSV, JSON,
Markdown). Reports are generated on demand from `backend/reports/generator.py`
— nothing is written to disk here, `st.download_button` serves the
bytes directly.
"""

from __future__ import annotations

import streamlit as st

from app.components.metric_card import MetricCardData, render_metric_row, section_header
from backend.experiments.storage import list_runs
from backend.reports.generator import (
    generate_comparison_csv,
    generate_comparison_json,
    generate_comparison_markdown,
    generate_csv,
    generate_json,
    generate_markdown,
    generate_pdf,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _render_single_run_tab(runs) -> None:
    labels = {f"{r.config.name}  ·  {r.dataset_name}  ·  {r.created_at[:19]}": r.id for r in runs}
    chosen = st.selectbox("Experiment run", options=list(labels.keys()), key="report_single_run")
    run = next(r for r in runs if r.id == labels[chosen])

    render_metric_row(
        [
            MetricCardData(label="Overall Score", value=f"{run.overall_score():.2f}", tone="info"),
            MetricCardData(label="Items", value=str(len(run.items)), tone="info"),
            MetricCardData(label="Avg. Latency", value=f"{run.avg_latency_ms:.0f} ms", tone="info"),
            MetricCardData(label="Total Cost", value=f"${run.total_cost:.5f}", tone="info"),
        ]
    )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("**Download this run's report**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            "⬇ CSV",
            data=generate_csv(run),
            file_name=f"{run.config.name.replace(' ', '_')}_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇ JSON",
            data=generate_json(run),
            file_name=f"{run.config.name.replace(' ', '_')}_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "⬇ Markdown",
            data=generate_markdown(run),
            file_name=f"{run.config.name.replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col4:
        try:
            pdf_bytes = generate_pdf(run)
            st.download_button(
                "⬇ PDF",
                data=pdf_bytes,
                file_name=f"{run.config.name.replace(' ', '_')}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.button("⬇ PDF", disabled=True, use_container_width=True, help="Install fpdf2 (`pip install fpdf2`) to enable PDF export.")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    with st.expander("Preview Markdown report"):
        st.markdown(generate_markdown(run))


def _render_comparison_tab(runs) -> None:
    labels = {f"{r.config.name}  ·  {r.dataset_name}  ·  {r.created_at[:19]}": r.id for r in runs}
    chosen = st.multiselect("Experiment runs to compare (2 or more)", options=list(labels.keys()), key="report_comparison_runs")

    if len(chosen) < 2:
        st.caption("Select at least two runs to generate a comparison report.")
        return

    selected_runs = [r for r in runs if r.id in {labels[c] for c in chosen}]
    best = max(selected_runs, key=lambda r: r.overall_score())
    st.success(f"🏆 Best overall: **{best.config.name}** ({best.overall_score():.3f})")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇ Comparison CSV",
            data=generate_comparison_csv(selected_runs),
            file_name="experiment_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇ Comparison JSON",
            data=generate_comparison_json(selected_runs),
            file_name="experiment_comparison.json",
            mime="application/json",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "⬇ Comparison Markdown",
            data=generate_comparison_markdown(selected_runs),
            file_name="experiment_comparison.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with st.expander("Preview Markdown comparison report"):
        st.markdown(generate_comparison_markdown(selected_runs))


def render() -> None:
    """Entry point called by `app/main.py` when 'Reports' is selected."""
    section_header("Reports", "Export any saved experiment run as CSV, JSON, Markdown, or PDF.")

    runs = list_runs()
    if not runs:
        st.info("No saved experiment runs yet — run one or more configurations in **Experiment Manager** first.")
        return

    tab_single, tab_compare = st.tabs(["Single run report", "Comparison report"])
    with tab_single:
        _render_single_run_tab(runs)
    with tab_compare:
        _render_comparison_tab(runs)
