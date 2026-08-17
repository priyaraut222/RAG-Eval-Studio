"""
Turns an `ExperimentRun` (or a list of them, for a comparison report)
into downloadable CSV, JSON, Markdown, or PDF bytes.

Every function here is pure — it takes data in and returns
str/bytes out — so the Reports page just wires a run/format choice to
a `st.download_button` and never touches formatting logic directly.

`generate_pdf` imports `fpdf2` lazily so the rest of this module (and
the app) stays importable even in an environment where that package
isn't installed — the Reports page catches the resulting `ImportError`
and tells the user which extra to install.
"""

from __future__ import annotations

import csv
import io
import json

from backend.experiments.results import ExperimentRun

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


# --------------------------------------------------------------------------
# Single-run reports
# --------------------------------------------------------------------------


def generate_json(run: ExperimentRun) -> str:
    """Full-fidelity JSON export — the same shape used for on-disk persistence."""
    return run.model_dump_json(indent=2)


def generate_csv(run: ExperimentRun) -> str:
    """Per-item CSV: one row per dataset item, every metric as a column."""
    buffer = io.StringIO()
    fieldnames = (
        ["question", "ground_truth", "generated_answer", "retrieved_context"]
        + list(_RETRIEVAL_LABELS.keys())
        + list(_LLM_LABELS.keys())
        + ["latency_ms", "input_tokens", "output_tokens", "estimated_cost"]
    )
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for item in run.items:
        row = {
            "question": item.question,
            "ground_truth": item.ground_truth,
            "generated_answer": item.generated_answer,
            "retrieved_context": item.retrieved_context,
            "latency_ms": round(item.latency_ms, 2),
            "input_tokens": item.input_tokens,
            "output_tokens": item.output_tokens,
            "estimated_cost": round(item.estimated_cost, 6),
        }
        for key in _RETRIEVAL_LABELS:
            row[key] = round(item.retrieval_metrics.get(key, 0.0), 4)
        for key in _LLM_LABELS:
            row[key] = round(item.llm_scores.get(key, 0.0), 4)
        writer.writerow(row)
    return buffer.getvalue()


def generate_markdown(run: ExperimentRun) -> str:
    """Human-readable Markdown report: config, aggregates, best/worst, per-item detail."""
    lines: list[str] = []
    lines.append(f"# Evaluation Report — {run.config.name}")
    lines.append("")
    lines.append(f"- **Dataset:** {run.dataset_name}")
    lines.append(f"- **Generated:** {run.created_at}")
    lines.append(f"- **Configuration:** {run.config.summary()}")
    lines.append(f"- **Overall score:** {run.overall_score():.3f}")
    lines.append(f"- **Total cost:** ${run.total_cost:.6f}")
    lines.append(f"- **Avg. latency:** {run.avg_latency_ms:.1f} ms")
    lines.append("")

    lines.append("## Retrieval Metrics")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    for key, label in _RETRIEVAL_LABELS.items():
        if key in run.aggregate_retrieval:
            lines.append(f"| {label} | {run.aggregate_retrieval[key]:.3f} |")
    lines.append("")

    lines.append("## LLM Evaluation Metrics")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    for key, label in _LLM_LABELS.items():
        if key in run.aggregate_llm:
            lines.append(f"| {label} | {run.aggregate_llm[key]:.3f} |")
    lines.append("")

    lines.append("## Best Performing Questions")
    lines.append("")
    for item in run.best_items[:3]:
        lines.append(f"- **{item.question}** — score {item.overall_score():.2f}")
    lines.append("")

    lines.append("## Worst Performing Questions")
    lines.append("")
    for item in run.worst_items[:3]:
        lines.append(f"- **{item.question}** — score {item.overall_score():.2f}")
    lines.append("")

    lines.append("## Per-Question Detail")
    lines.append("")
    for item in run.items:
        lines.append(f"### {item.question}")
        lines.append("")
        lines.append(f"- **Ground truth:** {item.ground_truth}")
        lines.append(f"- **Generated answer:** {item.generated_answer}")
        lines.append(f"- **Overall score:** {item.overall_score():.2f}")
        metric_bits = ", ".join(f"{_LLM_LABELS.get(k, k)}={v:.2f}" for k, v in item.llm_scores.items())
        lines.append(f"- **LLM metrics:** {metric_bits}")
        lines.append("")

    return "\n".join(lines)


def generate_pdf(run: ExperimentRun) -> bytes:
    """One-page-plus summary PDF: config, aggregate metrics, best/worst questions."""
    from fpdf import FPDF  # deferred — only required when a PDF is actually requested

    def clean(text: str) -> str:
        # Core PDF fonts are latin-1 only; replace anything else rather than crashing
        # on smart quotes/em-dashes that often show up in LLM-generated text.
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, clean(f"RAG Evaluation Report: {run.config.name}"))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 6, clean(f"Dataset: {run.dataset_name}  |  Generated: {run.created_at[:19]}"))
    pdf.multi_cell(0, 6, clean(f"Configuration: {run.config.summary()}"))
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, clean(f"Overall score: {run.overall_score():.3f}"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, clean(f"Total cost: ${run.total_cost:.6f}   Avg. latency: {run.avg_latency_ms:.1f} ms"), ln=True)
    pdf.ln(4)

    def metric_table(title: str, rows: dict[str, float], labels: dict[str, str]) -> None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, clean(title), ln=True)
        pdf.set_font("Helvetica", "", 10)
        for key, label in labels.items():
            if key in rows:
                pdf.cell(0, 6, clean(f"  {label}: {rows[key]:.3f}"), ln=True)
        pdf.ln(2)

    metric_table("Retrieval Metrics", run.aggregate_retrieval, _RETRIEVAL_LABELS)
    metric_table("LLM Evaluation Metrics", run.aggregate_llm, _LLM_LABELS)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Worst Performing Questions", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for item in run.worst_items[:5]:
        pdf.multi_cell(0, 6, clean(f"  - [{item.overall_score():.2f}] {item.question}"))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Best Performing Questions", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for item in run.best_items[:5]:
        pdf.multi_cell(0, 6, clean(f"  - [{item.overall_score():.2f}] {item.question}"))

    output = pdf.output()
    return bytes(output)


# --------------------------------------------------------------------------
# Comparison reports (multiple runs)
# --------------------------------------------------------------------------


def generate_comparison_csv(runs: list[ExperimentRun]) -> str:
    buffer = io.StringIO()
    fieldnames = (
        ["configuration", "overall_score"]
        + list(_RETRIEVAL_LABELS.keys())
        + list(_LLM_LABELS.keys())
        + ["avg_latency_ms", "total_cost"]
    )
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for run in runs:
        row = {
            "configuration": run.config.name,
            "overall_score": round(run.overall_score(), 4),
            "avg_latency_ms": round(run.avg_latency_ms, 2),
            "total_cost": round(run.total_cost, 6),
        }
        for key in _RETRIEVAL_LABELS:
            row[key] = round(run.aggregate_retrieval.get(key, 0.0), 4)
        for key in _LLM_LABELS:
            row[key] = round(run.aggregate_llm.get(key, 0.0), 4)
        writer.writerow(row)
    return buffer.getvalue()


def generate_comparison_json(runs: list[ExperimentRun]) -> str:
    return json.dumps([json.loads(r.model_dump_json()) for r in runs], indent=2)


def generate_comparison_markdown(runs: list[ExperimentRun]) -> str:
    lines: list[str] = ["# Experiment Comparison Report", ""]
    if runs:
        lines.append(f"Dataset: **{runs[0].dataset_name}**  ")
    lines.append(f"Configurations compared: {len(runs)}")
    lines.append("")

    headers = ["Configuration", "Overall"] + list(_RETRIEVAL_LABELS.values()) + list(_LLM_LABELS.values()) + ["Latency (ms)", "Cost ($)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for run in runs:
        cells = [run.config.name, f"{run.overall_score():.3f}"]
        cells += [f"{run.aggregate_retrieval.get(k, 0.0):.3f}" for k in _RETRIEVAL_LABELS]
        cells += [f"{run.aggregate_llm.get(k, 0.0):.3f}" for k in _LLM_LABELS]
        cells += [f"{run.avg_latency_ms:.1f}", f"{run.total_cost:.6f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    if runs:
        best = max(runs, key=lambda r: r.overall_score())
        lines.append(f"**Best overall configuration:** {best.config.name} ({best.overall_score():.3f}) — {best.config.summary()}")
        lines.append("")

    return "\n".join(lines)
