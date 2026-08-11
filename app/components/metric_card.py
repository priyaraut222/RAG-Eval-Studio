"""
Reusable UI components: metric cards, badges, and section headers.

These render raw HTML into `.rag-card` containers styled by
`app/components/styles.py`, so every page gets a consistent look
without duplicating markup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st

BadgeTone = Literal["success", "warning", "danger", "info"]


def _tone_for_score(score: float, higher_is_better: bool = True) -> BadgeTone:
    """Map a 0-1 score to a semantic tone for coloring."""
    normalized = score if higher_is_better else 1 - score
    if normalized >= 0.8:
        return "success"
    if normalized >= 0.5:
        return "warning"
    return "danger"


@dataclass
class MetricCardData:
    """Data needed to render one metric summary card."""

    label: str
    value: str
    delta: str | None = None
    tone: BadgeTone = "info"
    help_text: str | None = None


def render_metric_card(data: MetricCardData) -> None:
    """Render a single metric as a styled card with an optional delta badge."""
    delta_html = ""
    if data.delta:
        delta_html = f'<span class="rag-badge rag-badge-{data.tone}">{data.delta}</span>'

    help_html = ""
    if data.help_text:
        help_html = f'<div style="color: var(--rag-text-muted); font-size: 0.78rem; margin-top: 0.35rem;">{data.help_text}</div>'

    st.markdown(
        f"""
        <div class="rag-card">
            <div style="color: var(--rag-text-secondary); font-size: 0.85rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.03em;">
                {data.label}
            </div>
            <div style="display:flex; align-items:baseline; gap:0.5rem; margin-top:0.35rem;">
                <span style="font-size: 1.9rem; font-weight: 700; color: var(--rag-text-primary);">
                    {data.value}
                </span>
                {delta_html}
            </div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(cards: list[MetricCardData]) -> None:
    """Lay out a list of `MetricCardData` in an even-width row of columns."""
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            render_metric_card(card)


def render_score_card(label: str, score: float, help_text: str | None = None, higher_is_better: bool = True) -> None:
    """Convenience wrapper for the common case: a 0-1 metric score."""
    tone = _tone_for_score(score, higher_is_better)
    render_metric_card(
        MetricCardData(
            label=label,
            value=f"{score:.2f}",
            delta=tone.upper(),
            tone=tone,
            help_text=help_text,
        )
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a consistent page/section header."""
    subtitle_html = (
        f'<div style="color: var(--rag-text-secondary); margin-top: 0.15rem;">{subtitle}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.4rem; font-weight: 700; color: var(--rag-text-primary);">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
