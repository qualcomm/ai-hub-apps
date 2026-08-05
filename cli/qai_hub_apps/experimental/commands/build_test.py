# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qai_hub_apps.configs.app_yaml import AppLanguage, AppType
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.conftest import make_app_info
from qai_hub_apps.errors import InvalidArgumentError, QAIHubAppsError
from qai_hub_apps.experimental.commands import build as build_mod
from qai_hub_apps.experimental.commands.build import (
    _build_command,
    _prepare_app,
    _resolve_app_from_dir,
    run_build,
)
from qai_hub_apps.registry.base import App, Registry


@pytest.fixture
def sample_app_dir(tmp_path) -> Callable[[App], Path]:
    """Factory: create a fetched-app dir (info.yaml + build.sh + build.ps1) for an App."""

    def _make(app: App) -> Path:
        app_dir = tmp_path / app.id
        app_dir.mkdir(parents=True, exist_ok=True)
        (app_dir / "info.yaml").write_text(f"id: {app.id}\n", encoding="utf-8")
        (app_dir / "build.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (app_dir / "build.ps1").write_text("", encoding="utf-8")
        return app_dir

    return _make


def test_resolve_app_from_dir_reads_id(sample_app_dir, sample_registry_yaml):
    registry = Registry.load(sample_registry_yaml)
    app_dir = sample_app_dir(registry.find_by_id("test_app"))
    app = _resolve_app_from_dir(app_dir, registry)
    assert app.id == "test_app"


def test_resolve_app_from_dir_missing_info_raises(tmp_path, sample_registry_yaml):
    (tmp_path / "app").mkdir()
    with pytest.raises(InvalidArgumentError, match="Could not read an app 'id'"):
        _resolve_app_from_dir(tmp_path / "app", Registry.load(sample_registry_yaml))


def test_resolve_app_from_dir_malformed_info_raises(tmp_path, sample_registry_yaml):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "info.yaml").write_text("not a mapping\n", encoding="utf-8")
    with pytest.raises(InvalidArgumentError, match="Could not read an app 'id'"):
        _resolve_app_from_dir(app_dir, Registry.load(sample_registry_yaml))


def test_prepare_app_path_builds_in_place(
    tmp_path, sample_app_dir, sample_registry_yaml
):
    registry = Registry.load(sample_registry_yaml)
    app_dir = sample_app_dir(registry.find_by_id("test_app"))
    app, resolved = _prepare_app(None, app_dir, tmp_path, registry, None)
    assert resolved == app_dir.resolve()
    assert app.id == "test_app"


def test_prepare_app_path_warns_on_model(
    tmp_path, sample_app_dir, sample_registry_yaml, caplog
):
    registry = Registry.load(sample_registry_yaml)
    app_dir = sample_app_dir(registry.find_by_id("test_app"))
    _prepare_app(None, app_dir, tmp_path, registry, ModelAsset(model_id="m"))
    assert "ignored when building from a path" in caplog.text


def test_prepare_app_id_reuses_existing_dir(
    tmp_path, sample_app_dir, sample_registry_yaml, monkeypatch
):
    registry = Registry.load(sample_registry_yaml)
    app_dir = sample_app_dir(registry.find_by_id("test_app"))
    run_fetch = MagicMock()
    monkeypatch.setattr(build_mod, "run_fetch", run_fetch)
    _, resolved = _prepare_app("test_app", None, tmp_path, registry, None)
    run_fetch.assert_not_called()
    assert resolved == app_dir


def test_prepare_app_id_reuse_warns_on_model(
    tmp_path, sample_app_dir, sample_registry_yaml, caplog, monkeypatch
):
    registry = Registry.load(sample_registry_yaml)
    sample_app_dir(registry.find_by_id("test_app"))
    monkeypatch.setattr(build_mod, "run_fetch", MagicMock())
    _prepare_app("test_app", None, tmp_path, registry, ModelAsset(model_id="m"))
    assert "reusing an existing app" in caplog.text


def test_prepare_app_id_fetches_when_absent(
    tmp_path, sample_registry_yaml, monkeypatch
):
    fetched = tmp_path / "test_app"
    run_fetch = MagicMock(return_value=fetched)
    monkeypatch.setattr(build_mod, "run_fetch", run_fetch)
    _, app_dir = _prepare_app(
        "test_app",
        None,
        tmp_path,
        Registry.load(sample_registry_yaml),
        ModelAsset(model_id="m"),
    )
    run_fetch.assert_called_once()
    assert app_dir == fetched


def test_prepare_app_fetch_without_model_raises(
    tmp_path, sample_registry_yaml, monkeypatch
):
    monkeypatch.setattr(build_mod, "run_fetch", MagicMock())
    with pytest.raises(InvalidArgumentError, match="requires fetching"):
        _prepare_app(
            "test_app", None, tmp_path, Registry.load(sample_registry_yaml), None
        )


def test_prepare_app_disable_model_fetch_no_model_ok(tmp_path, monkeypatch):
    registry = MagicMock()
    registry.find_by_id.return_value = App(make_app_info(disable_cli_model_fetch=True))
    fetched = tmp_path / "test_app"
    monkeypatch.setattr(build_mod, "run_fetch", MagicMock(return_value=fetched))
    _, app_dir = _prepare_app("test_app", None, tmp_path, registry, None)
    assert app_dir == fetched


def test_prepare_app_overwrite_fetches_over_existing(
    tmp_path, sample_app_dir, sample_registry_yaml, monkeypatch
):
    registry = Registry.load(sample_registry_yaml)
    fetched = sample_app_dir(registry.find_by_id("test_app"))
    run_fetch = MagicMock(return_value=fetched)
    monkeypatch.setattr(build_mod, "run_fetch", run_fetch)
    _prepare_app(
        "test_app",
        None,
        tmp_path,
        registry,
        ModelAsset(model_id="m"),
        overwrite=True,
    )
    run_fetch.assert_called_once()
    assert run_fetch.call_args.kwargs["overwrite"] is True


def test_build_command_ubuntu_bash(sample_app_dir):
    app = App(make_app_info(app_type=AppType.UBUNTU))
    app_dir = sample_app_dir(app)
    cmd = _build_command(app, app_dir, use_docker=True, clean=False)
    assert cmd == ["bash", str(app_dir / "build.sh")]


def test_build_command_windows_powershell(sample_app_dir):
    app = App(make_app_info(app_type=AppType.WINDOWS, languages=[AppLanguage.CPP]))
    app_dir = sample_app_dir(app)
    cmd = _build_command(app, app_dir, use_docker=True, clean=False)
    assert cmd == ["powershell", "-File", str(app_dir / "build.ps1")]


def test_build_command_appends_flags(sample_app_dir):
    app = App(make_app_info(app_type=AppType.UBUNTU))
    cmd = _build_command(app, sample_app_dir(app), use_docker=False, clean=True)
    assert "--no-docker" in cmd
    assert "--clean" in cmd


def test_build_command_windows_flags(sample_app_dir):
    app = App(make_app_info(app_type=AppType.WINDOWS, languages=[AppLanguage.CPP]))
    cmd = _build_command(app, sample_app_dir(app), use_docker=False, clean=True)
    assert "-NoDocker" in cmd
    assert "-Clean" in cmd


def test_build_command_missing_script_raises(tmp_path):
    app = App(make_app_info(app_type=AppType.UBUNTU))
    with pytest.raises(QAIHubAppsError, match="No build script found"):
        _build_command(app, tmp_path, use_docker=True, clean=False)


def test_run_build_happy_path(tmp_path, sample_registry_yaml, monkeypatch):
    app = Registry.load(sample_registry_yaml).find_by_id("test_app")
    command = ["bash", str(tmp_path / "build.sh"), "--no-docker"]
    monkeypatch.setattr(
        build_mod, "_prepare_app", MagicMock(return_value=(app, tmp_path))
    )
    monkeypatch.setattr(build_mod, "ensure_build_supported", MagicMock())
    monkeypatch.setattr(build_mod, "_build_command", MagicMock(return_value=command))
    run = MagicMock()
    monkeypatch.setattr(build_mod.subprocess, "run", run)
    run_build("test_app", None, tmp_path, MagicMock(), None, use_docker=False)
    run.assert_called_once()
    assert run.call_args.args[0] == command
    assert run.call_args.kwargs["cwd"] == tmp_path


def test_run_build_subprocess_failure_raises(
    tmp_path, sample_registry_yaml, monkeypatch
):
    app = Registry.load(sample_registry_yaml).find_by_id("test_app")
    monkeypatch.setattr(
        build_mod, "_prepare_app", MagicMock(return_value=(app, tmp_path))
    )
    monkeypatch.setattr(build_mod, "ensure_build_supported", MagicMock())
    monkeypatch.setattr(
        build_mod, "_build_command", MagicMock(return_value=["bash", "build.sh"])
    )
    monkeypatch.setattr(
        build_mod.subprocess,
        "run",
        MagicMock(side_effect=subprocess.CalledProcessError(2, "bash")),
    )
    with pytest.raises(QAIHubAppsError, match="Build failed"):
        run_build("test_app", None, tmp_path, MagicMock(), None, use_docker=True)
