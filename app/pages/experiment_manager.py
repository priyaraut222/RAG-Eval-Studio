"""
Experiment Manager page.

Flow: pick a dataset + source text → define one or more named
configurations (chunk size, retriever, vector store, Top-K, reranker,
LLM provider) → run all of them → see a side-by-side comparison table
with the best configuration per metric highlighted, plus a bar chart
and per-run detail. Runs are persisted so the "Saved Experiments" tab
can compare across sessions.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.metric_card import MetricCardData, render_metric_row, section_header
from backend.dataset.storage import list_datasets, load_dataset
from backend.experiments.config import EMBEDDING_MODEL_CHOICES, ExperimentConfig
from backend.experiments.results import ExperimentRun
from backend.experiments.runner import run_experiment
from backend.experiments.storage import delete_run, list_runs, save_run
from backend.utils.logger import get_logger

logger = get_logger(__name__)

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


def _init_state() -> None:
    st.session_state.setdefault("experiment_configs", [])  # list[ExperimentConfig]
    st.session_state.setdefault("experiment_source_text", "")
    st.session_state.setdefault("experiment_runs", [])  # list[ExperimentRun], this session


def _render_source_panel() -> tuple | None:
    datasets = list_datasets()
    if not datasets:
        st.info("No saved datasets yet — build one in **Dataset Builder** first.")
        return None

    dataset_labels = {f"{d.name}  ·  {d.size} items": d.id for d in datasets}
    chosen_label = st.selectbox("Dataset", options=list(dataset_labels.keys()))
    dataset_id = dataset_labels[chosen_label]
    dataset = load_dataset(dataset_id)

    col1, col2 = st.columns([5, 1])
    with col1:
        source_text = st.text_area(
            "Source text to index",
            value=st.session_state["experiment_source_text"],
            height=120,
            placeholder="Paste the source document text every configuration will be evaluated against...",
        )
        st.session_state["experiment_source_text"] = source_text
    with col2:
        st.markdown("<div style='height:1.9rem;'></div>", unsafe_allow_html=True)
        if st.button("↺ Reuse from Retrieval Eval", help="Copy the source text from the Retrieval Evaluation page, if you've pasted one there."):
            reused = st.session_state.get("retrieval_source_text", "")
            if reused:
                st.session_state["experiment_source_text"] = reused
                st.rerun()
            else:
                st.warning("No source text found on the Retrieval Evaluation page yet.")

    return dataset, source_text


def _render_config_builder() -> None:
    section_header("Configurations", "Define the RAG pipeline variants you want to compare side by side.")

    with st.expander("➕ Add a configuration", expanded=not st.session_state["experiment_configs"]):
        name = st.text_input("Configuration name", placeholder="e.g. Small chunks + reranker", key="new_cfg_name")

        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_size = st.slider("Chunk size", 200, 2000, 800, step=100, key="new_cfg_chunk_size")
            chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, step=20, key="new_cfg_chunk_overlap")
        with col2:
            retriever = st.selectbox("Retriever", options=["tfidf", "embedding"], key="new_cfg_retriever")
            top_k = st.slider("Top-K", 1, 10, 5, key="new_cfg_top_k")
        with col3:
            llm_provider = st.selectbox("LLM provider", options=["openai", "gemini", "local"], index=2, key="new_cfg_provider")
            use_reranker = st.checkbox("Use reranker", key="new_cfg_reranker")

        embedding_model = EMBEDDING_MODEL_CHOICES[0]
        vector_store = "in_memory"
        if retriever == "embedding":
            col4, col5 = st.columns(2)
            with col4:
                embedding_model = st.selectbox("Embedding model", options=EMBEDDING_MODEL_CHOICES, key="new_cfg_embedding_model")
            with col5:
                vector_store = st.selectbox(
                    "Vector store",
                    options=["in_memory", "faiss", "chroma"],
                    key="new_cfg_vector_store",
                    help="Falls back to in-memory automatically if faiss-cpu/chromadb aren't installed.",
                )

        if st.button("Add configuration", type="primary", disabled=not name.strip()):
            config = ExperimentConfig(
                name=name.strip(),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                retriever=retriever,
                embedding_model=embedding_model,
                vector_store=vector_store,
                top_k=top_k,
                use_reranker=use_reranker,
                llm_provider=llm_provider,
            )
            st.session_state["experiment_configs"].append(config)
            st.rerun()

    configs: list[ExperimentConfig] = st.session_state["experiment_configs"]
    if not configs:
        st.caption("No configurations added yet.")
        return

    for config in configs:
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(
                f"<div class='rag-card' style='margin-bottom:0.5rem;'>"
                f"<strong>{config.name}</strong><br>"
                f"<span style='color:var(--rag-text-secondary); font-size:0.85rem;'>{config.summary()}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("Remove", key=f"remove_cfg_{config.id}"):
                st.session_state["experiment_configs"] = [c for c in configs if c.id != config.id]
                st.rerun()


def _render_run_panel(dataset, source_text: str) -> None:
    configs: list[ExperimentConfig] = st.session_state["experiment_configs"]
    if not configs:
        return

    if st.button(f"▶ Run all {len(configs)} configuration(s)", type="primary", disabled=not source_text.strip()):
        results: list[ExperimentRun] = []
        progress = st.progress(0.0, text="Starting...")
        for i, config in enumerate(configs):
            progress.progress(i / len(configs), text=f"Running '{config.name}'...")
            try:
                run = run_experiment(dataset, config, source_text)
                results.append(run)
                save_run(run)
            except ValueError as exc:
                st.error(f"'{config.name}' failed: {exc}")
        progress.progress(1.0, text="Done.")

        st.session_state["experiment_runs"] = results
        st.success(f"Completed {len(results)} run(s). See the comparison below and the **Saved Experiments** tab.")


def _highlight_best(df: pd.DataFrame, higher_is_better_cols: list[str]) -> "pd.io.formats.style.Styler":
    def _style_col(col: pd.Series):
        if col.name not in higher_is_better_cols or col.empty:
            return ["" for _ in col]
        best = col.max()
        return ["background-color: var(--rag-primary-soft); font-weight: 700;" if v == best else "" for v in col]

    return df.style.apply(_style_col, axis=0)


def _render_comparison(runs: list[ExperimentRun]) -> None:
    if not runs:
        return

    section_header("Comparison", f"{len(runs)} configuration(s) evaluated against the same dataset.")

    rows = []
    for run in runs:
        row = {"Configuration": run.config.name, "Overall": round(run.overall_score(), 3)}
        for key, label in _RETRIEVAL_LABELS.items():
            row[label] = round(run.aggregate_retrieval.get(key, 0.0), 3)
        for key, label in _LLM_LABELS.items():
            row[label] = round(run.aggregate_llm.get(key, 0.0), 3)
        row["Latency (ms)"] = round(run.avg_latency_ms, 1)
        row["Cost ($)"] = round(run.total_cost, 5)
        rows.append(row)

    df = pd.DataFrame(rows)
    higher_is_better = ["Overall"] + list(_RETRIEVAL_LABELS.values()) + [l for l in _LLM_LABELS.values() if l != "Hallucination"]
    st.dataframe(_highlight_best(df, higher_is_better), use_container_width=True, height=min(80 + 40 * len(rows), 400))

    best = max(runs, key=lambda r: r.overall_score())
    st.markdown(
        f"<div class='rag-card' style='margin-top:0.75rem;'>"
        f"🏆 <strong>{best.config.name}</strong> has the best overall score "
        f"(<span class='rag-badge rag-badge-success'>{best.overall_score():.2f}</span>) — {best.config.summary()}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    fig = go.Figure()
    for run in runs:
        fig.add_trace(go.Bar(name=run.config.name, x=["Overall", "Retrieval nDCG@K", "Faithfulness", "Answer Correctness"],
                              y=[run.overall_score(), run.aggregate_retrieval.get("ndcg_at_k", 0), run.aggregate_llm.get("faithfulness", 0), run.aggregate_llm.get("answer_correctness", 0)]))
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, 1.05]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_saved_tab() -> None:
    runs = list_runs()
    if not runs:
        st.info("No saved experiment runs yet — run a comparison above.")
        return

    _render_comparison(runs)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("**Manage saved runs**")
    for run in runs:
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.caption(f"{run.config.name} · {run.dataset_name} · {run.created_at[:19]}")
        with col2:
            render_metric_row([MetricCardData(label="Overall", value=f"{run.overall_score():.2f}", tone="info")])
        with col3:
            if st.button("Delete", key=f"delete_run_{run.id}"):
                delete_run(run.id)
                st.rerun()


def render() -> None:
    """Entry point called by `app/main.py` when 'Experiment Manager' is selected."""
    _init_state()
    section_header(
        "Experiment Manager",
        "Define multiple RAG pipeline configurations, run them against the same dataset, and see which wins.",
    )

    source = _render_source_panel()
    if source is None:
        return
    dataset, source_text = source

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    _render_config_builder()

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    _render_run_panel(dataset, source_text)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    tab_current, tab_saved = st.tabs(["This session's runs", "Saved Experiments"])
    with tab_current:
        _render_comparison(st.session_state["experiment_runs"])
        if not st.session_state["experiment_runs"]:
            st.caption("Run configurations above to see a comparison here.")
    with tab_saved:
        _render_saved_tab()
