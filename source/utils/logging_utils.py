"""Logging setup shared by the API and the worker."""
from __future__ import annotations

import logging
import os

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once (level from ``PS2_LOG_LEVEL``, default INFO)."""
    level = level or os.environ.get("PS2_LOG_LEVEL", "INFO")
    logging.basicConfig(level=level.upper() if isinstance(level, str) else level,
                        format=LOG_FORMAT)
