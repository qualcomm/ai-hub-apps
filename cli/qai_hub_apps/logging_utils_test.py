# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging

import pytest

from qai_hub_apps.logging_utils import LOG_LEVEL_ENV_VAR, configure_logging


@pytest.fixture
def fresh_logger():
    """Provide the package logger and root handlers with a clean slate."""
    logger = logging.getLogger("qai_hub_apps")
    saved_level = logger.level
    root = logging.getLogger()
    saved_root_handlers = root.handlers[:]
    yield logger
    logger.setLevel(saved_level)
    root.handlers[:] = saved_root_handlers


@pytest.mark.parametrize(
    ("flag", "env", "expected"),
    [
        ("debug", None, logging.DEBUG),  # flag
        (None, None, logging.INFO),  # default
        (None, "error", logging.ERROR),  # env var
        ("debug", "error", logging.DEBUG),  # flag overrides env var
        ("DEBUG", None, logging.DEBUG),  # case-insensitive
        ("bogus", None, logging.INFO),  # unknown falls back to default
    ],
)
def test_resolves_level(fresh_logger, monkeypatch, flag, env, expected):
    if env is None:
        monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(LOG_LEVEL_ENV_VAR, env)
    configure_logging(flag)
    assert fresh_logger.level == expected


def test_unknown_level_warns(fresh_logger, monkeypatch, caplog):
    monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
    with caplog.at_level(logging.WARNING, logger="qai_hub_apps"):
        configure_logging("bogus")
    assert "Unknown log level" in caplog.text
    assert "bogus" in caplog.text


def test_installs_single_root_handler(fresh_logger):
    logging.getLogger().handlers.clear()
    configure_logging("info")
    configure_logging("debug")  # second call is a no-op for basicConfig
    assert len(logging.getLogger().handlers) == 1
