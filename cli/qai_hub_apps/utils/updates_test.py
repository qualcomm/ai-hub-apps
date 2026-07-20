# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging

from qai_hub_apps.utils import updates


def test_notice_when_newer_available(monkeypatch, caplog):
    monkeypatch.setattr(updates, "__version__", "1.0.0")
    monkeypatch.setattr(
        updates, "_get_latest_version", lambda: updates.Version("2.0.0")
    )
    with caplog.at_level(logging.INFO, logger="qai_hub_apps"):
        updates.check_for_update()
    assert "newer version" in caplog.text
    assert "2.0.0" in caplog.text


def test_no_notice_when_up_to_date(monkeypatch, caplog):
    monkeypatch.setattr(updates, "__version__", "1.0.0")
    monkeypatch.setattr(
        updates, "_get_latest_version", lambda: updates.Version("1.0.0")
    )
    with caplog.at_level(logging.INFO, logger="qai_hub_apps"):
        updates.check_for_update()
    assert "newer version" not in caplog.text


def test_errors_are_swallowed(monkeypatch):
    def _boom() -> updates.Version:
        raise OSError("no network")

    monkeypatch.setattr(updates, "_get_latest_version", _boom)
    updates.check_for_update()  # must not raise
