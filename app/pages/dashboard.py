"""
Dashboard page.

Pulls together whatever's been run this session in Retrieval
Evaluation and/or LLM Evaluation (`st.session_state["retrieval_report"]`
/ `st.session_state["llm_eval_report"]`) into one view: an overall
score, a metric-card grid, a radar chart spanning both metric
families, latency/cost graphs, and a combined best/worst questions
table. Persistent cross-session experiment comparison is Phase 6.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.metric_card import MetricCardData, render_metric_row, section_header

_RETRIEVAL_LABELS = {
    "precision_at_k": "Precision@K",
    "recall_at_k": "Recall@K",
    "mrr": "MRR",
    "hit_rate": "Hit Rate",
    "ndcg_at_k": "nDCG@K",
}
_LLM_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "hallucination": "Hallucination",
    "answer_correctness": "Answer Correctness",
}


def _get_reports():
    return st.session_state.get("retrieval_report"), st.session_state.get("llm_eval_report")


def _overall_score(retrieval_report, llm_report) -> float | None:
    """Blend retrieval + LLM scores into one headline number.

    Hallucination is excluded (inverted scale) — its signal already
    lives inside the LLM metric cards and the radar chart.
    """
    parts: list[float] = []
    if retrieval_report is not None:
        parts.extend(v for k, v in retrieval_report.aggregate.items() if k != "mrr" or True)
    if llm_report is not None:
        parts.extend(v for k, v in llm_report.aggregate.items() if k != "hallucination")
    if not parts:
        return None
    return sum(parts) / len(parts)


def _render_overall_and_cards(retrieval_report, llm_report) -> None:
    overall = _overall_score(retrieval_report, llm_report)
    cards = [
        MetricCardData(
            label="Overall Evaluation Score",
            value=f"{overall:.2f}" if overall is not None else "—",
            tone="success" if (overall or 0) >= 0.7 else "warning" if (overall or 0) >= 0.4 else "danger",
            help_text="Blended average across retrieval + generation metrics (excl. Hallucination)",
        )
    ]
    if retrieval_report is not None:
        cards.append(MetricCardData(label="Retrieval nDCG@K", value=f"{retrieval_report.aggregate.get('ndcg_at_k', 0):.2f}", tone="info"))
    if llm_report is not None:
        cards.append(MetricCardData(label="Faithfulness", value=f"{llm_report.aggregate.get('faithfulness', 0):.2f}", tone="info"))
        cards.append(MetricCardData(label="Estimated Cost", value=f"${llm_report.total_cost:.4f}", tone="info"))
    render_metric_row(cards)


def _render_radar(retrieval_report, llm_report) -> None:
    labels: list[str] = []
    values: list[float] = []

    if retrieval_report is not None:
        for key, label in _RETRIEVAL_LABELS.items():
            labels.append(label)
            values.append(retrieval_report.aggregate.get(key, 0.0))
    if llm_report is not None:
        for key, label in _LLM_LABELS.items():
            labels.append(label)
            # Flip hallucination so the radar reads "outward = better" consistently.
            score = llm_report.aggregate.get(key, 0.0)
            values.append(1 - score if key == "hallucination" else score)

    if not labels:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line_color="#4F63D2",
            fillcolor="rgba(79, 99, 210, 0.25)",
            name="Current run",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=420,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_latency_cost_graphs(llm_report) -> None:
    if llm_report is None or not llm_report.per_item:
        st.info("Run **LLM Evaluation** to see latency and cost graphs.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Latency per question**")
        labels = [f"Q{i+1}" for i in range(len(llm_report.per_item))]
        latencies = [r.latency_ms for r in llm_report.per_item]
        fig = go.Figure(go.Bar(x=labels, y=latencies, marker_color="#2FB6A1"))
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="ms",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Cumulative estimated cost**")
        costs = [r.estimated_cost for r in llm_report.per_item]
        cumulative = []
        running = 0.0
        for c in costs:
            running += c
            cumulative.append(running)
        labels = [f"Q{i+1}" for i in range(len(llm_report.per_item))]
        fig = go.Figure(go.Scatter(x=labels, y=cumulative, mode="lines+markers", line_color="#E8A33D"))
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="$",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_best_worst_table(retrieval_report, llm_report) -> None:
    rows = []
    if llm_report is not None:
        for r in llm_report.per_item:
            rows.append(
                {
                    "Question": r.question,
                    "Source": "LLM Eval",
                    "Score": round(r.overall_score(), 2),
                }
            )
    if retrieval_report is not None:
        for r in retrieval_report.per_item:
            rows.append(
                {
                    "Question": r.question,
                    "Source": "Retrieval",
                    "Score": round(r.metrics.get("ndcg_at_k", 0.0), 2),
                }
            )

    if not rows:
        st.info("Run an evaluation to populate best/worst performing questions.")
        return

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏆 Best performing**")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**⚠️ Worst performing**")
        st.dataframe(df.tail(5).sort_values("Score"), use_container_width=True, hide_index=True)


def render() -> None:
    """Entry point called by `app/main.py` when 'Dashboard' is selected."""
    section_header(
        "Dashboard",
        "A unified view of retrieval and generation quality for the current session's evaluation runs.",
    )

    retrieval_report, llm_report = _get_reports()
    if retrieval_report is None and llm_report is None:
        st.info(
            "No evaluation runs yet this session. Run **Retrieval Evaluation** and/or **LLM Evaluation** "
            "first — this dashboard updates automatically from those results."
        )
        return

    _render_overall_and_cards(retrieval_report, llm_report)
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        section_header("Metric radar", None)
        _render_radar(retrieval_report, llm_report)
    with col2:
        section_header("Best / worst questions", None)
        _render_best_worst_table(retrieval_report, llm_report)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    section_header("Latency & cost", None)
    _render_latency_cost_graphs(llm_report)
