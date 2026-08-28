# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qai_hub_apps import user_config
from qai_hub_apps.conftest import make_device
from qai_hub_apps.errors import QAIHubAppsError


@pytest.fixture
def config_file(tmp_path, monkeypatch) -> Path:
    """Point the CLI config at a temp file (not created) and return its path."""
    path = tmp_path / "config" / "config.json"
    monkeypatch.setattr(user_config, "config_path", lambda: path)
    return path


def test_config_path_under_user_config_dir(monkeypatch):
    monkeypatch.setattr(user_config, "user_config_dir", lambda _: "/cfg/qai-hub-apps")
    assert user_config.config_path() == Path("/cfg/qai-hub-apps/config.json")


def test_get_configured_device_none_when_not_a_string(config_file):
    config_file.parent.mkdir(parents=True)
    config_file.write_text('{"device": 5}', encoding="utf-8")
    assert user_config.get_configured_device() is None


def test_get_configured_device_resolves_name(config_file, monkeypatch):
    device = make_device(name="Device A")
    resolve = MagicMock(return_value=device)
    monkeypatch.setattr(user_config, "resolve_device_info", resolve)
    config_file.parent.mkdir(parents=True)
    config_file.write_text('{"device": "Device A"}', encoding="utf-8")
    assert user_config.get_configured_device() is device
    resolve.assert_called_once_with("Device A")


def test_set_configured_device_preserves_other_keys(config_file):
    config_file.parent.mkdir(parents=True)
    config_file.write_text('{"other": 1}', encoding="utf-8")
    assert (
        user_config.set_configured_device(make_device(name="Device B")) == config_file
    )
    assert json.loads(config_file.read_text(encoding="utf-8")) == {
        "other": 1,
        "device": "Device B",
    }


def test_set_configured_device_write_failure_raises(config_file, monkeypatch):
    monkeypatch.setattr(
        user_config.Path, "mkdir", MagicMock(side_effect=OSError("denied"))
    )
    with pytest.raises(QAIHubAppsError, match="Could not write config"):
        user_config.set_configured_device(make_device())
