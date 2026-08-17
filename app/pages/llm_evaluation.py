"""
LLM Evaluation page.

Flow: pick a saved dataset → pick which field to treat as "retrieved
context" (expected_context or expected_chunk, since Phase 4 doesn't
require a live external retriever) → run → see six metric cards, a
per-metric explainability panel (what/why/how), a per-question table,
and an error-analysis view for the worst-scoring items.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.metric_card import MetricCardData, render_metric_row, section_header
from backend.config.settings import get_settings
from backend.dataset.storage import list_datasets, load_dataset
from backend.evaluation.metrics.registry import get_all_metrics
from backend.evaluation.runner import LLMEvalReport, run_llm_evaluation
from backend.utils.llm_client import get_llm_client
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "hallucination": "Hallucination",
    "answer_correctness": "Answer Correctness",
}


def _init_state() -> None:
    st.session_state.setdefault("llm_eval_report", None)


def _tone_for(metric_key: str, score: float) -> str:
    """Hallucination is inverted — high is bad — so its tone logic flips."""
    bad = score if metric_key == "hallucination" else 1 - score
    if bad <= 0.2:
        return "success"
    if bad <= 0.5:
        return "warning"
    return "danger"


def _render_config_panel():
    datasets = list_datasets()
    if not datasets:
        st.info("No saved datasets yet — build one in **Dataset Builder** first.")
        return None

    dataset_labels = {f"{d.name}  ·  {d.size} items": d.id for d in datasets}
    chosen_label = st.selectbox("Dataset", options=list(dataset_labels.keys()))
    dataset_id = dataset_labels[chosen_label]
    dataset = load_dataset(dataset_id)

    settings = get_settings()
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox(
            "LLM provider",
            options=["openai", "gemini", "local"],
            index=["openai", "gemini", "local"].index(settings.default_llm_provider),
            help="If the selected provider has no API key configured, this automatically falls back "
            "to a deterministic offline generator/judge so the page stays fully demoable.",
        )
    with col2:
        context_source = st.selectbox(
            "Context source",
            options=["expected_context", "expected_chunk"],
            format_func=lambda v: "Expected Context" if v == "expected_context" else "Expected Chunk",
            help="Which dataset field to treat as the 'retrieved context' for generation and scoring, "
            "since this run doesn't require a live external retriever.",
        )

    if st.button("▶ Run LLM evaluation", type="primary"):
        settings.default_llm_provider = provider  # type: ignore[assignment]
        client = get_llm_client(settings)

        if client.active_provider != provider:
            st.warning(
                f"No API key configured for **{provider}** — using the offline heuristic generator/judge "
                "instead. Add a key in `.env` for real LLM-quality generation and scoring."
            )

        with st.spinner(f"Generating answers and scoring with {client.active_provider}..."):
            report = run_llm_evaluation(dataset, client, context_source=context_source)

        if not report.per_item:
            st.error("No complete items to evaluate — items need a question, ground truth, and expected context/chunk.")
            return st.session_state["llm_eval_report"]

        st.session_state["llm_eval_report"] = report
        st.success(f"Evaluated {len(report.per_item)} item(s) with {report.provider}/{report.model}.")

    return st.session_state["llm_eval_report"]


def _render_metric_cards(report: LLMEvalReport) -> None:
    cards = []
    for key, label in _METRIC_LABELS.items():
        score = report.aggregate.get(key, 0.0)
        tone = _tone_for(key, score)
        cards.append(MetricCardData(label=label, value=f"{score:.2f}", delta=tone.upper(), tone=tone))
    render_metric_row(cards[:3])
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    render_metric_row(cards[3:])


def _render_cost_latency_cards(report: LLMEvalReport) -> None:
    render_metric_row(
        [
            MetricCardData(label="Avg. Latency", value=f"{report.avg_latency_ms:.0f} ms", tone="info"),
            MetricCardData(label="Total Tokens", value=f"{report.total_input_tokens + report.total_output_tokens:,}", tone="info"),
            MetricCardData(label="Estimated Cost", value=f"${report.total_cost:.4f}", tone="info", help_text="Indicative, not billing-accurate — see backend/config/pricing.py"),
            MetricCardData(label="Items Evaluated", value=str(len(report.per_item)), tone="info"),
        ]
    )


def _render_explainability_panel() -> None:
    st.markdown("**How each metric is calculated**")
    for metric in get_all_metrics():
        suffix = " (lower is better)" if not metric.higher_is_better else ""
        with st.expander(f"{metric.label}{suffix}"):
            st.markdown(f"**What it means:** {metric.explainer.what}")
            st.markdown(f"**Why it matters:** {metric.explainer.why}")
            st.markdown(f"**How it's calculated:** {metric.explainer.how}")


def _render_results_table(report: LLMEvalReport) -> None:
    rows = []
    for r in report.per_item:
        row = {"Question": r.question, "Generated Answer": r.generated_answer[:80]}
        for key, label in _METRIC_LABELS.items():
            row[label] = round(r.metrics[key].score, 2) if key in r.metrics else None
        row["Latency (ms)"] = round(r.latency_ms, 1)
        row["Cost ($)"] = round(r.estimated_cost, 5)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=380)


def _render_error_analysis(report: LLMEvalReport) -> None:
    worst = [r for r in report.worst_items if r.overall_score() < 0.6][:5]
    if not worst:
        st.success("No items scored below the 0.6 overall-score threshold — nothing flagged for review.")
        return

    suggestions = {
        "faithfulness": "Tighten the generation prompt to require answers to only use the given context.",
        "answer_relevancy": "Check whether the question is ambiguous, or the answer is padded with unrelated detail.",
        "context_precision": "The retriever may be returning too much irrelevant text — try a smaller chunk size or a reranker.",
        "context_recall": "The retriever likely missed needed information — try a larger Top-K, different chunking, or a stronger embedding model.",
        "answer_correctness": "The generator may be misreading the context, or the context itself may be ambiguous/insufficient.",
    }

    for r in worst:
        with st.expander(f"⚠️ {r.question}  ·  overall {r.overall_score():.2f}"):
            st.markdown(f"**Ground truth:** {r.ground_truth}")
            st.markdown(f"**Retrieved context:** {r.context[:400]}{'…' if len(r.context) > 400 else ''}")
            st.markdown(f"**Generated answer:** {r.generated_answer}")
            st.markdown("**Scores:**")
            score_cols = st.columns(len(r.metrics))
            for col, (key, m) in zip(score_cols, r.metrics.items()):
                with col:
                    st.metric(_METRIC_LABELS.get(key, key), f"{m.score:.2f}")
            lowest_key = min(
                (k for k in r.metrics if k != "hallucination"),
                key=lambda k: r.metrics[k].score,
                default=None,
            )
            if lowest_key:
                st.caption(f"**Likely cause:** lowest score is *{_METRIC_LABELS[lowest_key]}* — {r.metrics[lowest_key].reasoning}")
                if lowest_key in suggestions:
                    st.caption(f"**Suggested improvement:** {suggestions[lowest_key]}")


def render() -> None:
    """Entry point called by `app/main.py` when 'LLM Evaluation' is selected."""
    _init_state()
    section_header(
        "LLM Evaluation",
        "Faithfulness, Answer Relevancy, Context Precision/Recall, Hallucination, and Answer Correctness.",
    )

    report = _render_config_panel()
    if report is None:
        return

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    section_header("Results", f"{report.provider}/{report.model} · context source: {report.context_source}")
    _render_metric_cards(report)
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    _render_cost_latency_cards(report)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    tab_table, tab_errors, tab_explain = st.tabs(["Per-question results", "Error analysis", "Explainability"])
    with tab_table:
        _render_results_table(report)
    with tab_errors:
        _render_error_analysis(report)
    with tab_explain:
        _render_explainability_panel()
