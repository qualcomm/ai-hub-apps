# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qai_hub_apps.commands.fetch import run_fetch
from qai_hub_apps.errors import QAIHubAppsError


def _make_registry(fetch_return=Path("/tmp/test_app")):
    registry = MagicMock()
    registry.version = "1.0.0"
    registry.fetch_app.return_value = fetch_return
    return registry


def test_run_fetch_success(caplog, capsys):
    dest = Path("/tmp/test_app")
    registry = _make_registry(fetch_return=dest)
    with caplog.at_level("INFO", logger="qai_hub_apps"):
        run_fetch("test_app", Path("/tmp"), registry)
    registry.fetch_app.assert_called_once_with(
        "test_app", Path("/tmp"), model_asset=None
    )
    assert dest.as_posix() in caplog.text
    # The fetched path is printed to stdout so it can be piped to other commands.
    assert capsys.readouterr().out.strip() == str(dest)


def test_run_fetch_with_model_asset(capsys):
    from qai_hub_apps.configs.model_asset import ModelAsset

    registry = _make_registry()
    asset = ModelAsset(model_id="whisper_base", chipset=None)
    run_fetch("test_app", Path("/tmp"), registry, model_asset=asset)
    registry.fetch_app.assert_called_once_with(
        "test_app", Path("/tmp"), model_asset=asset
    )


def test_run_fetch_http_error_raises():
    registry = MagicMock()
    registry.version = "1.0.0"
    from unittest.mock import MagicMock as _MM

    registry.fetch_app.side_effect = urllib.error.HTTPError(
        url="https://example.com", code=404, msg="Not Found", hdrs=_MM(), fp=None
    )
    with pytest.raises(QAIHubAppsError, match="404"):
        run_fetch("test_app", Path("/tmp"), registry)


def test_run_fetch_url_error_raises():
    registry = MagicMock()
    registry.version = "1.0.0"
    registry.fetch_app.side_effect = urllib.error.URLError(reason="connection refused")
    with pytest.raises(QAIHubAppsError, match="connection refused"):
        run_fetch("test_app", Path("/tmp"), registry)
