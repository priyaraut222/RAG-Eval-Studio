"""
Turns the design tokens in `backend.config.theme` into a CSS string
that gets injected into the Streamlit app via `st.markdown(..., unsafe_allow_html=True)`.

Keeping this generation logic separate from `main.py` means every
page/component can call `inject_theme()` once and rely on a
consistent set of CSS custom properties (`--rag-*`) for the rest of
the app's HTML snippets (metric cards, badges, etc.).
"""

from __future__ import annotations

import streamlit as st

from backend.config.theme import (
    FONT_FAMILY,
    FONT_FAMILY_MONO,
    RADIUS_LG,
    RADIUS_MD,
    RADIUS_SM,
    get_palette,
)


def _build_css(mode: str) -> str:
    p = get_palette(mode)

    return f"""
    <style>
    :root {{
        --rag-background: {p.background};
        --rag-surface: {p.surface};
        --rag-surface-alt: {p.surface_alt};
        --rag-border: {p.border};
        --rag-text-primary: {p.text_primary};
        --rag-text-secondary: {p.text_secondary};
        --rag-text-muted: {p.text_muted};
        --rag-primary: {p.primary};
        --rag-primary-hover: {p.primary_hover};
        --rag-primary-soft: {p.primary_soft};
        --rag-success: {p.success};
        --rag-warning: {p.warning};
        --rag-danger: {p.danger};
        --rag-info: {p.info};
        --rag-radius-sm: {RADIUS_SM};
        --rag-radius-md: {RADIUS_MD};
        --rag-radius-lg: {RADIUS_LG};
        --rag-font: {FONT_FAMILY};
        --rag-font-mono: {FONT_FAMILY_MONO};
    }}

    html, body, [class*="css"] {{
        font-family: var(--rag-font);
    }}

    .stApp {{
        background-color: var(--rag-background);
        color: var(--rag-text-primary);
    }}

    section[data-testid="stSidebar"] {{
        background-color: var(--rag-surface);
        border-right: 1px solid var(--rag-border);
    }}

    /* Headings */
    h1, h2, h3, h4 {{
        color: var(--rag-text-primary);
        font-weight: 650;
        letter-spacing: -0.01em;
    }}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {{
        background-color: var(--rag-primary);
        color: white;
        border: none;
        border-radius: var(--rag-radius-sm);
        padding: 0.5rem 1.1rem;
        font-weight: 600;
        transition: background-color 0.15s ease, transform 0.1s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: var(--rag-primary-hover);
        transform: translateY(-1px);
    }}

    /* Generic card container used by metric_card.py, etc. */
    .rag-card {{
        background-color: var(--rag-surface);
        border: 1px solid var(--rag-border);
        border-radius: var(--rag-radius-lg);
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }}
    .rag-card:hover {{
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.08);
        transform: translateY(-1px);
    }}

    .rag-badge {{
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .rag-badge-success {{ background: color-mix(in srgb, var(--rag-success) 18%, transparent); color: var(--rag-success); }}
    .rag-badge-warning {{ background: color-mix(in srgb, var(--rag-warning) 18%, transparent); color: var(--rag-warning); }}
    .rag-badge-danger  {{ background: color-mix(in srgb, var(--rag-danger) 18%, transparent);  color: var(--rag-danger); }}
    .rag-badge-info    {{ background: color-mix(in srgb, var(--rag-info) 18%, transparent);    color: var(--rag-info); }}

    /* Tables */
    .stDataFrame {{
        border-radius: var(--rag-radius-md);
        overflow: hidden;
        border: 1px solid var(--rag-border);
    }}

    /* Tighter default top padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* Smooth fade-in for main content */
    .main .block-container {{
        animation: rag-fade-in 0.25s ease;
    }}
    @keyframes rag-fade-in {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """


def inject_theme() -> None:
    """Inject the current theme's CSS into the page.

    Reads `st.session_state["theme_mode"]` (set by the sidebar toggle
    in `main.py`), defaulting to light mode on first load.
    """
    mode = st.session_state.get("theme_mode", "light")
    st.markdown(_build_css(mode), unsafe_allow_html=True)
