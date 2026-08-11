"""
Design tokens for RAG Evaluation Studio.

A single source of truth for color, spacing, and typography so the
Streamlit CSS injection and Plotly chart theming never drift apart.
Light mode is the default; dark mode is an explicit, user-selected
alternate palette (never a pure-black background).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Palette:
    """A named color palette for one theme mode."""

    name: str

    # Surfaces
    background: str
    surface: str
    surface_alt: str
    border: str

    # Text
    text_primary: str
    text_secondary: str
    text_muted: str

    # Brand / accent
    primary: str
    primary_hover: str
    primary_soft: str

    # Semantic status colors (used consistently across metric cards/charts)
    success: str
    warning: str
    danger: str
    info: str

    # Chart categorical sequence (colorblind-conscious, muted-professional)
    chart_sequence: tuple[str, ...] = field(
        default_factory=lambda: (
            "#4F63D2",  # indigo
            "#2FB6A1",  # teal
            "#E8A33D",  # amber
            "#D25B7A",  # rose
            "#6C7FDB",  # periwinkle
            "#3D9970",  # green
        )
    )


LIGHT = Palette(
    name="light",
    background="#F7F8FB",
    surface="#FFFFFF",
    surface_alt="#F0F2F8",
    border="#E3E6EF",
    text_primary="#1B1F2B",
    text_secondary="#4A5068",
    text_muted="#8A90A6",
    primary="#4F63D2",
    primary_hover="#3E4FB8",
    primary_soft="#EDEFFC",
    success="#1FA97F",
    warning="#E8A33D",
    danger="#D2495C",
    info="#3E8FD0",
)

# Dark mode: deep slate/navy surfaces — deliberately NOT pure black —
# so cards, shadows, and dividers stay legible.
DARK = Palette(
    name="dark",
    background="#161A24",
    surface="#1E2333",
    surface_alt="#252B3D",
    border="#323A52",
    text_primary="#EDEFF7",
    text_secondary="#B7BCD1",
    text_muted="#7C8299",
    primary="#7C8CF0",
    primary_hover="#93A1F5",
    primary_soft="#2A2F4D",
    success="#3ECF98",
    warning="#F0B94D",
    danger="#E8697D",
    info="#5FA8E8",
)

FONT_FAMILY = (
    '"Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif'
)
FONT_FAMILY_MONO = '"JetBrains Mono", "Fira Code", monospace'

RADIUS_SM = "8px"
RADIUS_MD = "12px"
RADIUS_LG = "18px"

SPACE_XS = "4px"
SPACE_SM = "8px"
SPACE_MD = "16px"
SPACE_LG = "24px"
SPACE_XL = "32px"


def get_palette(mode: str) -> Palette:
    """Return the palette for `mode` ('light' or 'dark'), defaulting to light."""
    return DARK if mode == "dark" else LIGHT
