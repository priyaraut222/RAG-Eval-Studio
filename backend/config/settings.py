"""
Application-wide settings for RAG Evaluation Studio.

All configuration is centralized here so the rest of the codebase
never reaches into `os.environ` directly. Values are loaded from a
`.env` file (see `.env.example`) with sane, explicit defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (backend/config/settings.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Instantiate once via `get_settings()` — do not construct this
    class directly elsewhere in the app.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App metadata ---
    app_name: str = "RAG Evaluation Studio"
    app_version: str = "0.1.0"
    environment: Literal["development", "production", "test"] = "development"

    # --- LLM provider credentials (optional at import time; validated on use) ---
    openai_api_key: str | None = Field(default=None)
    google_api_key: str | None = Field(default=None)

    # --- Default model choices ---
    default_llm_provider: Literal["openai", "gemini", "local"] = "openai"
    default_openai_model: str = "gpt-4o-mini"
    default_gemini_model: str = "gemini-1.5-flash"
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval defaults ---
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64
    default_top_k: int = 5

    # --- Vector store ---
    default_vector_store: Literal["faiss", "chroma"] = "faiss"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_stores"

    # --- Storage paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    datasets_dir: Path = PROJECT_ROOT / "data" / "datasets"
    experiments_dir: Path = PROJECT_ROOT / "data" / "experiments"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"
    logs_dir: Path = PROJECT_ROOT / "data" / "logs"

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def ensure_directories(self) -> None:
        """Create all storage directories if they don't exist yet."""
        for directory in (
            self.data_dir,
            self.datasets_dir,
            self.experiments_dir,
            self.reports_dir,
            self.logs_dir,
            self.vector_store_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached singleton `Settings` instance.

    Using a module-level cache (instead of `@lru_cache` directly on a
    free function importing this) keeps the pattern simple and mock-
    friendly in tests, where `_settings` can be reset directly.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
