# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qai_hub_apps.commands.list_apps import run_info, run_list
from qai_hub_apps.configs.app_yaml import AppLanguage
from qai_hub_apps.conftest import make_app_info
from qai_hub_apps.errors import AppNotFoundError
from qai_hub_apps.registry.base import App


def _make_registry_with_apps(*infos):
    registry = MagicMock()
    registry.apps = [App(info) for info in infos]
    return registry


def test_run_list_prints_app_count(capsys):
    registry = _make_registry_with_apps(
        make_app_info(id="app_a", name="App A"),
        make_app_info(id="app_b", name="App B"),
    )
    run_list(registry)
    out = capsys.readouterr().out
    assert "Total: 2 apps" in out


def test_run_list_happy_path_output(capsys):
    registry = _make_registry_with_apps(
        make_app_info(
            id="whisper_windows_py",
            name="Whisper",
            domain="Audio",
            languages=[AppLanguage.PYTHON, AppLanguage.CPP],
        ),
        make_app_info(id="stable_diffusion_py", name="Stable Diffusion"),
    )
    run_list(registry)
    out = capsys.readouterr().out
    assert "whisper_windows_py" in out
    assert "stable_diffusion_py" in out

    assert "Domain" in out
    assert "Languages" in out
    assert "Audio" in out
    assert "Python, C++" in out


def test_run_list_empty_registry(capsys):
    registry = _make_registry_with_apps()
    run_list(registry)
    out = capsys.readouterr().out
    assert "0 apps" in out


def test_run_info_prints_app_repr(capsys):
    info = make_app_info(id="test_app", name="Test App")
    registry = MagicMock()
    app = App(info)
    registry.find_by_id.return_value = app
    run_info("test_app", registry)
    registry.find_by_id.assert_called_once_with("test_app")
    out = capsys.readouterr().out
    assert app.__repr__() in out


def test_run_info_not_found_propagates():
    registry = MagicMock()
    registry.find_by_id.side_effect = AppNotFoundError("missing_app")
    with pytest.raises(AppNotFoundError):
        run_info("missing_app", registry)


def test_run_info_valid_id_from_registry(sample_registry_yaml, capsys):
    from qai_hub_apps.registry.base import Registry

    registry = Registry.load(sample_registry_yaml)
    run_info("test_app", registry)
    out = capsys.readouterr().out
    assert "Test App" in out


def test_run_info_invalid_id_from_registry_raises(sample_registry_yaml):
    from qai_hub_apps.registry.base import Registry

    registry = Registry.load(sample_registry_yaml)
    with pytest.raises(AppNotFoundError, match="nonexistent_app"):
        run_info("nonexistent_app", registry)
