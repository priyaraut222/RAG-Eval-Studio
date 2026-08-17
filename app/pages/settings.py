"""
Settings page.

Shows the effective configuration (from `.env` via `backend.config.settings`)
and lets the person override the default LLM provider/models and
retrieval defaults for the rest of this session. Actual API keys are
never entered here — they live in `.env` (see `.env.example`) and are
only ever shown masked, since Streamlit session state isn't a secret
store.
"""

from __future__ import annotations

import streamlit as st

from app.components.metric_card import section_header
from backend.config.settings import get_settings
from backend.experiments.config import EMBEDDING_MODEL_CHOICES


def _mask(key: str | None) -> str:
    if not key:
        return "Not configured"
    return f"{'•' * max(len(key) - 4, 4)}{key[-4:]}"


def _render_key_status() -> None:
    settings = get_settings()
    section_header("API Keys", "Configured via `.env` — never entered in the browser.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="rag-card">
                <div style="font-weight:700;">OpenAI</div>
                <div style="color:var(--rag-text-secondary); font-family:var(--rag-font-mono); margin-top:0.3rem;">
                    {_mask(settings.openai_api_key)}
                </div>
                <div style="margin-top:0.4rem;">
                    <span class="rag-badge {'rag-badge-success' if settings.openai_api_key else 'rag-badge-warning'}">
                        {'Configured' if settings.openai_api_key else 'Not set — falls back to offline heuristic'}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="rag-card">
                <div style="font-weight:700;">Google Gemini</div>
                <div style="color:var(--rag-text-secondary); font-family:var(--rag-font-mono); margin-top:0.3rem;">
                    {_mask(settings.google_api_key)}
                </div>
                <div style="margin-top:0.4rem;">
                    <span class="rag-badge {'rag-badge-success' if settings.google_api_key else 'rag-badge-warning'}">
                        {'Configured' if settings.google_api_key else 'Not set — falls back to offline heuristic'}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "To add keys: copy `.env.example` to `.env` in the project root, fill in "
        "`OPENAI_API_KEY` and/or `GOOGLE_API_KEY`, and restart the app. Without either key, "
        "every page still works end-to-end using the built-in offline heuristic provider."
    )


def _render_defaults() -> None:
    settings = get_settings()
    section_header("Session Defaults", "Pre-fills the provider/model choices on other pages for the rest of this session.")

    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox(
            "Default LLM provider",
            options=["openai", "gemini", "local"],
            index=["openai", "gemini", "local"].index(settings.default_llm_provider),
        )
        st.session_state["default_llm_provider_override"] = provider
    with col2:
        embedding_model = st.selectbox(
            "Default embedding model",
            options=EMBEDDING_MODEL_CHOICES,
            index=EMBEDDING_MODEL_CHOICES.index(settings.default_embedding_model) if settings.default_embedding_model in EMBEDDING_MODEL_CHOICES else 0,
        )
        st.session_state["default_embedding_model_override"] = embedding_model

    col3, col4, col5 = st.columns(3)
    with col3:
        st.number_input("Default chunk size", min_value=200, max_value=2000, value=settings.default_chunk_size, step=100, key="default_chunk_size_override")
    with col4:
        st.number_input("Default chunk overlap", min_value=0, max_value=400, value=settings.default_chunk_overlap, step=20, key="default_chunk_overlap_override")
    with col5:
        st.number_input("Default Top-K", min_value=1, max_value=10, value=settings.default_top_k, key="default_top_k_override")

    st.caption("These apply for the current browser session only — they don't rewrite your `.env` file.")


def _render_about() -> None:
    settings = get_settings()
    section_header("About")
    st.markdown(
        f"""
        <div class="rag-card">
            <div><strong>{settings.app_name}</strong> · v{settings.app_version}</div>
            <div style="color:var(--rag-text-secondary); margin-top:0.4rem; font-size:0.9rem;">
                Environment: {settings.environment}<br>
                Data directory: <code>{settings.data_dir}</code><br>
                Default retriever backend: TF-IDF (numpy) with optional Sentence-Transformers embeddings,
                FAISS, and ChromaDB — all with automatic graceful fallback if not installed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    """Entry point called by `app/main.py` when 'Settings' is selected."""
    section_header("Settings", "Provider configuration, defaults, and app info.")
    _render_key_status()
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    _render_defaults()
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    _render_about()
