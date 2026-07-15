# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
import os

LOG_LEVEL_ENV_VAR = "QAI_HUB_APPS_LOG_LEVEL"

DEFAULT_LEVEL = "info"

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "error": logging.ERROR,
}

_PACKAGE_LOGGER_NAME = "qai_hub_apps"


def set_log_level(level: str | None = None) -> None:
    """Set the ``qai_hub_apps`` logger level.

    Parameters
    ----------
    level:
        One of ``"debug"``, ``"info"``, ``"error"`` (case-insensitive), or
        None to fall back to the ``QAI_HUB_APPS_LOG_LEVEL`` env var, then the
        default (``"info"``).
    """
    # Precedence: flag > env var > default; an unknown value falls back to default.
    requested = level or os.environ.get(LOG_LEVEL_ENV_VAR)
    numeric_level = _LEVELS.get(
        (requested or DEFAULT_LEVEL).lower(), _LEVELS[DEFAULT_LEVEL]
    )

    logging.getLogger(_PACKAGE_LOGGER_NAME).setLevel(numeric_level)

    if requested is not None and requested.lower() not in _LEVELS:
        logging.getLogger(_PACKAGE_LOGGER_NAME).warning(
            "Unknown log level %r; falling back to %r.", requested, DEFAULT_LEVEL
        )


def is_quiet() -> bool:
    """Return True when the CLI is running in quiet mode (``-q``/``--quiet``).

    Quiet mode raises the ``qai_hub_apps`` logger level to ERROR, so anything
    at ERROR or above (i.e. INFO/DEBUG suppressed) is treated as quiet. Used to
    silence third-party output such as download progress bars.
    """
    return logging.getLogger(_PACKAGE_LOGGER_NAME).getEffectiveLevel() >= logging.ERROR


def configure_logging(level: str | None = None) -> None:
    """Set the ``qai_hub_apps`` log level and send diagnostics to stderr.

    Use ``logging.basicConfig`` beforehand for custom formatting.

    Parameters
    ----------
    level:
        One of ``"debug"``, ``"info"``, ``"error"`` (case-insensitive), or
        None to fall back to the ``QAI_HUB_APPS_LOG_LEVEL`` env var, then the
        default (``"info"``).
    """
    set_log_level(level)
    logging.basicConfig()
