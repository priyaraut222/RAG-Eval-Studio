"""
Centralized logging for RAG Evaluation Studio.

Every module should obtain its logger via `get_logger(__name__)`
rather than configuring `loguru`/`logging` independently. This keeps
log format, level, and sinks consistent across the app and backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from backend.config.settings import get_settings

_configured = False


def _configure() -> None:
    """Configure loguru sinks exactly once per process."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    _logger.remove()  # drop the default stderr sink so we control format

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    )

    _logger.add(sys.stderr, level=settings.log_level, format=log_format, colorize=True)

    log_file: Path = settings.logs_dir / "rag_eval_studio.log"
    _logger.add(
        log_file,
        level="DEBUG",
        format=log_format,
        rotation="5 MB",
        retention="14 days",
        colorize=False,
    )

    _configured = True


def get_logger(name: str):
    """Return a loguru logger bound with a module `name` tag.

    Usage:
        from backend.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    _configure()
    return _logger.bind(module=name)
