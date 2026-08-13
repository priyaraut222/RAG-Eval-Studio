"""
Retrieval Evaluation page.

Flow: pick a saved dataset → supply the source text it was built from
(or reuse what's in the Dataset Builder session) → pick a retriever +
chunking/Top-K config → run → see per-item metrics, a ranking chart,
and best/worst performing questions.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.metric_card import MetricCardData, render_metric_row, section_header
from backend.dataset.storage import list_datasets, load_dataset
from backend.retrieval.evaluator import RetrievalEvalReport, evaluate_dataset_retrieval
from backend.retrieval.retriever import get_retriever
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _init_state() -> None:
    st.session_state.setdefault("retrieval_report", None)
    st.session_state.setdefault("retrieval_source_text", "")


def _render_config_panel() -> tuple | None:
    datasets = list_datasets()
    if not datasets:
        st.info("No saved datasets yet — build one in **Dataset Builder** first.")
        return None

    dataset_labels = {f"{d.name}  ·  {d.size} items": d.id for d in datasets}
    chosen_label = st.selectbox("Dataset", options=list(dataset_labels.keys()))
    dataset_id = dataset_labels[chosen_label]
    dataset = load_dataset(dataset_id)

    source_text = st.text_area(
        "Source text to index",
        value=st.session_state["retrieval_source_text"],
        height=140,
        help="The corpus the retriever searches over — usually the same document(s) "
        "the dataset's questions were built from. Paste it here if it wasn't carried "
        "over from the Dataset Builder.",
        placeholder="Paste the source document text here...",
    )
    st.session_state["retrieval_source_text"] = source_text

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        retriever_name = st.selectbox("Retriever", options=["tfidf", "embedding"], help="TF-IDF runs fully offline. Embedding uses sentence-transformers if available.")
    with col2:
        top_k = st.slider("Top-K", 1, 10, 5)
    with col3:
        chunk_size = st.slider("Chunk size", 200, 2000, 800, step=100)
    with col4:
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, step=20)

    run = st.button("▶ Run retrieval evaluation", type="primary", disabled=not source_text.strip())
    if run:
        retriever = get_retriever(retriever_name)
        with st.spinner(f"Indexing and scoring with {retriever.name}..."):
            try:
                report = evaluate_dataset_retrieval(
                    dataset=dataset,
                    retriever=retriever,
                    source_text=source_text,
                    top_k=top_k,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except ValueError as exc:
                st.error(str(exc))
                return None
        st.session_state["retrieval_report"] = report
        if retriever.name != retriever_name:
            st.warning(f"Requested **{retriever_name}** retriever wasn't available — used **{retriever.name}** instead.")

    return st.session_state["retrieval_report"]


def _render_aggregate_cards(report: RetrievalEvalReport) -> None:
    agg = report.aggregate
    render_metric_row(
        [
            MetricCardData(label="Precision@K", value=f"{agg.get('precision_at_k', 0):.2f}", tone="info"),
            MetricCardData(label="Recall@K", value=f"{agg.get('recall_at_k', 0):.2f}", tone="info"),
            MetricCardData(label="MRR", value=f"{agg.get('mrr', 0):.2f}", tone="info"),
            MetricCardData(label="Hit Rate", value=f"{agg.get('hit_rate', 0):.2f}", tone="info"),
            MetricCardData(label="nDCG@K", value=f"{agg.get('ndcg_at_k', 0):.2f}", tone="info"),
        ]
    )


def _render_bar_chart(report: RetrievalEvalReport) -> None:
    agg = report.aggregate
    labels = ["Precision@K", "Recall@K", "MRR", "Hit Rate", "nDCG@K"]
    values = [agg.get("precision_at_k", 0), agg.get("recall_at_k", 0), agg.get("mrr", 0), agg.get("hit_rate", 0), agg.get("ndcg_at_k", 0)]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#4F63D2", text=[f"{v:.2f}" for v in values], textposition="outside"))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, 1.05], title="Score"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_ranking_table(report: RetrievalEvalReport) -> None:
    rows = []
    for r in report.per_item:
        rows.append(
            {
                "Question": r.question,
                "Precision@K": round(r.metrics["precision_at_k"], 2),
                "Recall@K": round(r.metrics["recall_at_k"], 2),
                "nDCG@K": round(r.metrics["ndcg_at_k"], 2),
                "Hit Rate": round(r.metrics["hit_rate"], 2),
                "Latency (ms)": round(r.latency_ms, 2),
                "Top Retrieved": r.retrieved[0].text[:80] + "…" if r.retrieved else "—",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=360)


def _render_best_worst(report: RetrievalEvalReport) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏆 Best performing questions**")
        for r in report.best_items[:3]:
            st.markdown(
                f"<div class='rag-card' style='margin-bottom:0.5rem;'>"
                f"<span class='rag-badge rag-badge-success'>nDCG {r.metrics['ndcg_at_k']:.2f}</span> "
                f"<span style='margin-left:0.4rem;'>{r.question}</span></div>",
                unsafe_allow_html=True,
            )
    with col2:
        st.markdown("**⚠️ Worst performing questions**")
        for r in report.worst_items[:3]:
            st.markdown(
                f"<div class='rag-card' style='margin-bottom:0.5rem;'>"
                f"<span class='rag-badge rag-badge-danger'>nDCG {r.metrics['ndcg_at_k']:.2f}</span> "
                f"<span style='margin-left:0.4rem;'>{r.question}</span></div>",
                unsafe_allow_html=True,
            )


def render() -> None:
    """Entry point called by `app/main.py` when 'Retrieval Evaluation' is selected."""
    _init_state()
    section_header(
        "Retrieval Evaluation",
        "Precision@K, Recall@K, MRR, Hit Rate, and nDCG for a dataset against a chosen retriever.",
    )

    report = _render_config_panel()
    if report is None:
        return

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    section_header("Results", f"{report.retriever_name} retriever · top-{report.top_k} · chunk size {report.chunk_size}")
    _render_aggregate_cards(report)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    _render_bar_chart(report)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    _render_best_worst(report)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("**Per-question detail**")
    _render_ranking_table(report)
