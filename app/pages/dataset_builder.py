"""
Dataset Builder page.

Lets a user:
1. Upload PDF / CSV / JSON source material
2. Generate a synthetic evaluation dataset (question, ground truth,
   expected context, expected chunk) via an LLM
3. Manually review and edit generated items in a data table
4. Save the result as a named `EvalDataset`, and browse/delete past ones

All parsing/generation/persistence logic lives in `backend/dataset/`;
this module is purely the UI layer that wires user input to it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.metric_card import section_header
from backend.config.settings import get_settings
from backend.dataset.loaders import (
    DatasetLoadError,
    load_csv_records,
    load_json_records,
    load_pdf_text,
    map_record_to_fields,
)
from backend.dataset.schema import EvalDataset, EvalItem
from backend.dataset.storage import delete_dataset, list_datasets, save_dataset
from backend.dataset.synthesizer import generate_synthetic_items
from backend.utils.llm_client import LLMError, get_llm_client
from backend.utils.logger import get_logger

logger = get_logger(__name__)

ITEM_COLUMNS = ["question", "ground_truth", "expected_context", "expected_chunk", "source_document", "source"]


def _init_state() -> None:
    st.session_state.setdefault("db_draft_items", [])  # list[dict] backing the editor
    st.session_state.setdefault("db_warnings", [])
    st.session_state.setdefault("db_source_texts", {})  # filename -> extracted text


def _items_to_dataframe(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=ITEM_COLUMNS)
    return pd.DataFrame(items)[ITEM_COLUMNS]


def _handle_uploads(uploaded_files) -> None:
    """Parse each uploaded file into either raw text (PDF) or mapped rows (CSV/JSON)."""
    new_rows: list[dict] = []
    warnings: list[str] = []

    for uploaded in uploaded_files:
        name = uploaded.name
        raw_bytes = uploaded.getvalue()
        suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""

        try:
            if suffix == "pdf":
                text = load_pdf_text(raw_bytes, filename=name)
                st.session_state["db_source_texts"][name] = text
            elif suffix == "csv":
                records = load_csv_records(raw_bytes, filename=name)
                for record in records:
                    mapped = map_record_to_fields(record)
                    new_rows.append(
                        {
                            "question": mapped.get("question", ""),
                            "ground_truth": mapped.get("ground_truth", ""),
                            "expected_context": mapped.get("expected_context", ""),
                            "expected_chunk": "",
                            "source_document": name,
                            "source": "imported",
                        }
                    )
            elif suffix == "json":
                records = load_json_records(raw_bytes, filename=name)
                for record in records:
                    mapped = map_record_to_fields(record)
                    new_rows.append(
                        {
                            "question": mapped.get("question", ""),
                            "ground_truth": mapped.get("ground_truth", ""),
                            "expected_context": mapped.get("expected_context", ""),
                            "expected_chunk": "",
                            "source_document": name,
                            "source": "imported",
                        }
                    )
            else:
                warnings.append(f"'{name}': unsupported file type '.{suffix}' — expected PDF, CSV, or JSON.")
        except DatasetLoadError as exc:
            warnings.append(str(exc))

    if new_rows:
        st.session_state["db_draft_items"].extend(new_rows)
    st.session_state["db_warnings"] = warnings


def _render_upload_section() -> None:
    section_header("1. Add source material", "Upload PDFs to synthesize questions from, or CSV/JSON files already shaped like Q&A pairs.")

    uploaded_files = st.file_uploader(
        "Upload PDF, CSV, or JSON",
        type=["pdf", "csv", "json"],
        accept_multiple_files=True,
        help="PDFs are parsed into source text for synthesis below. CSV/JSON rows are mapped onto question/ground_truth/context where possible.",
    )

    if uploaded_files and st.button("Process uploads", type="primary"):
        with st.spinner("Parsing uploaded files..."):
            _handle_uploads(uploaded_files)
        st.rerun()

    for warning in st.session_state["db_warnings"]:
        st.warning(warning)

    if st.session_state["db_source_texts"]:
        with st.expander(f"📄 Extracted source text ({len(st.session_state['db_source_texts'])} document(s))"):
            for name, text in st.session_state["db_source_texts"].items():
                st.markdown(f"**{name}** — {len(text):,} characters")
                st.text_area("Preview", value=text[:1500] + ("..." if len(text) > 1500 else ""), height=140, key=f"preview_{name}", disabled=True, label_visibility="collapsed")


def _render_synthesis_section() -> None:
    section_header("2. Generate synthetic questions", "Uses your configured LLM provider to draft question/answer pairs grounded in the uploaded PDF text.")

    settings = get_settings()
    source_texts = st.session_state["db_source_texts"]

    if not source_texts:
        st.caption("Upload at least one PDF above to enable synthetic generation.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=800, step=100)
    with col2:
        chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=100, step=50)
    with col3:
        questions_per_chunk = st.number_input("Questions / chunk", min_value=1, max_value=5, value=2, step=1)
    with col4:
        max_chunks = st.number_input("Max chunks / doc", min_value=1, max_value=100, value=20, step=1)

    st.caption(f"Provider: **{settings.default_llm_provider}** — change this in Settings.")

    if st.button("✨ Generate synthetic dataset", type="primary"):
        try:
            client = get_llm_client()
        except LLMError as exc:
            st.error(f"Can't generate yet: {exc}")
            return

        if client.active_provider != settings.default_llm_provider:
            st.warning(
                f"No API key configured for **{settings.default_llm_provider}** — using the offline "
                "heuristic generator instead. Add a key in `.env` (see Settings) for real LLM-quality questions."
            )

        progress = st.progress(0.0, text="Starting generation...")
        total_docs = len(source_texts)
        all_warnings: list[str] = []

        for i, (name, text) in enumerate(source_texts.items(), start=1):
            progress.progress((i - 1) / total_docs, text=f"Generating from {name} ({i}/{total_docs})...")
            items, warnings = generate_synthetic_items(
                source_text=text,
                source_document=name,
                client=client,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                questions_per_chunk=int(questions_per_chunk),
                max_chunks=int(max_chunks),
            )
            st.session_state["db_draft_items"].extend(item.model_dump() for item in items)
            all_warnings.extend(warnings)

        progress.progress(1.0, text="Done.")
        if all_warnings:
            with st.expander(f"⚠️ {len(all_warnings)} warning(s) during generation"):
                for warning in all_warnings:
                    st.caption(warning)
        st.success(f"Generated items from {total_docs} document(s). Review and edit them below.")
        st.rerun()


def _render_editor_section() -> EvalDataset | None:
    section_header("3. Review, edit, and save", "Every generated or imported item is editable. Incomplete rows (missing question or answer) are flagged before saving.")

    draft_items = st.session_state["db_draft_items"]
    df = _items_to_dataframe(draft_items)

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        height=380,
        column_config={
            "question": st.column_config.TextColumn("Question", width="medium"),
            "ground_truth": st.column_config.TextColumn("Ground Truth", width="medium"),
            "expected_context": st.column_config.TextColumn("Expected Context", width="large"),
            "expected_chunk": st.column_config.TextColumn("Expected Chunk", width="medium"),
            "source_document": st.column_config.TextColumn("Source", width="small"),
            "source": st.column_config.SelectboxColumn("Type", options=["manual", "synthetic", "imported"], width="small"),
        },
        key="db_editor",
    )

    # Persist edits back into session state so they survive reruns.
    st.session_state["db_draft_items"] = edited_df.fillna("").to_dict("records")

    incomplete = edited_df[(edited_df["question"].str.strip() == "") | (edited_df["ground_truth"].str.strip() == "")]
    complete_count = len(edited_df) - len(incomplete)

    left, right = st.columns([3, 1])
    with left:
        if len(incomplete) > 0:
            st.caption(f"⚠️ {len(incomplete)} row(s) missing a question or ground truth — these are excluded from saving.")
        st.caption(f"**{complete_count}** complete item(s) ready to save.")
    with right:
        if st.button("➕ Add blank row"):
            st.session_state["db_draft_items"].append(
                {"question": "", "ground_truth": "", "expected_context": "", "expected_chunk": "", "source_document": "manual", "source": "manual"}
            )
            st.rerun()

    st.divider()

    name = st.text_input("Dataset name", placeholder="e.g. Product Docs QA v1")
    description = st.text_area("Description (optional)", placeholder="What this dataset covers, how it was built, etc.", height=70)

    if st.button("💾 Save dataset", type="primary", disabled=(complete_count == 0 or not name.strip())):
        dataset = EvalDataset(name=name.strip(), description=description.strip())
        for record in edited_df.to_dict("records"):
            item = EvalItem(
                question=str(record.get("question", "")).strip(),
                ground_truth=str(record.get("ground_truth", "")).strip(),
                expected_context=str(record.get("expected_context", "")).strip(),
                expected_chunk=str(record.get("expected_chunk", "")).strip(),
                source_document=str(record.get("source_document", "")).strip(),
                source=record.get("source") or "manual",
            )
            if item.is_complete():
                dataset.add_item(item)

        save_dataset(dataset)
        st.session_state["db_draft_items"] = []
        st.session_state["db_source_texts"] = {}
        st.success(f"Saved '{dataset.name}' with {dataset.size} item(s).")
        st.rerun()

    return None


def _render_saved_datasets_section() -> None:
    section_header("Saved datasets", "Datasets available to Retrieval Evaluation, LLM Evaluation, and Experiments once those phases are built.")

    datasets = list_datasets()
    if not datasets:
        st.caption("No datasets saved yet.")
        return

    for dataset in datasets:
        with st.container():
            cols = st.columns([4, 2, 2, 1])
            cols[0].markdown(f"**{dataset.name}**  \n<span style='color:var(--rag-text-muted); font-size:0.8rem;'>{dataset.description or 'No description'}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"{dataset.size} items")
            cols[2].markdown(f"<span style='color:var(--rag-text-muted); font-size:0.8rem;'>{dataset.created_at[:10]}</span>", unsafe_allow_html=True)
            if cols[3].button("🗑️", key=f"delete_{dataset.id}", help="Delete this dataset"):
                delete_dataset(dataset.id)
                st.rerun()
        st.divider()


def render() -> None:
    """Entry point called by `app/main.py` when 'Dataset Builder' is selected."""
    _init_state()

    section_header("Dataset Builder", "Build the evaluation set your RAG configurations will be scored against.")

    tab_build, tab_saved = st.tabs(["Build a dataset", "Saved datasets"])
    with tab_build:
        _render_upload_section()
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        _render_synthesis_section()
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        _render_editor_section()
    with tab_saved:
        _render_saved_datasets_section()
