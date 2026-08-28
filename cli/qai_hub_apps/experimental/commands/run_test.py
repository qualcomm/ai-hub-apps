# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from qai_hub_apps.configs.app_yaml import AppLanguage, AppType
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.conftest import make_app_info, make_device
from qai_hub_apps.errors import QAIHubAppsError
from qai_hub_apps.experimental.commands import run as run_mod
from qai_hub_apps.experimental.commands.run import _run_command, run_run
from qai_hub_apps.registry.base import App

DEVICE = make_device(name="Device A")


def _make_app(app_type: AppType = AppType.UBUNTU, **overrides) -> App:
    languages = {
        AppType.WINDOWS: [AppLanguage.CPP],
        AppType.ANDROID: [AppLanguage.JAVA],
    }.get(app_type, [AppLanguage.PYTHON])
    return App(make_app_info(app_type=app_type, languages=languages, **overrides))


def _launch_script(app_dir: Path, name: str) -> Path:
    app_dir.mkdir(parents=True, exist_ok=True)
    script = app_dir / name
    script.write_text("", encoding="utf-8")
    return script


@pytest.fixture
def stub_run_run(monkeypatch) -> SimpleNamespace:
    """Stub out everything run_run calls; return the mocks by name."""
    mocks = SimpleNamespace(
        ensure_run_supported=MagicMock(),
        device_env=MagicMock(return_value={"DEV": "A"}),
        run_command=MagicMock(return_value=["bash", "x"]),
        subprocess_run=MagicMock(),
        run_build=MagicMock(return_value=Path("built")),
        get_configured_device=MagicMock(return_value=DEVICE),
    )
    monkeypatch.setattr(run_mod, "ensure_run_supported", mocks.ensure_run_supported)
    monkeypatch.setattr(run_mod, "device_env", mocks.device_env)
    monkeypatch.setattr(run_mod, "_run_command", mocks.run_command)
    monkeypatch.setattr(run_mod.subprocess, "run", mocks.subprocess_run)
    monkeypatch.setattr(run_mod, "run_build", mocks.run_build)
    monkeypatch.setattr(run_mod, "get_configured_device", mocks.get_configured_device)
    return mocks


def test_run_command_appends_flags_and_args(tmp_path):
    script = _launch_script(tmp_path, "launch.sh")
    command = _run_command(_make_app(), tmp_path, False, True, ["--verbose"], True)
    assert command == [
        "bash",
        str(script),
        "--no-docker",
        "--clean",
        "--test",
        "--",
        "--verbose",
    ]


def test_run_command_windows_flags(tmp_path):
    script = _launch_script(tmp_path, "launch.ps1")
    command = _run_command(_make_app(AppType.WINDOWS), tmp_path, False, True, [], True)
    assert command == [
        "powershell",
        "-File",
        str(script),
        "-NoDocker",
        "-Clean",
        "-Test",
    ]


def test_run_command_missing_script_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "_is_dev", lambda: False)
    with pytest.raises(QAIHubAppsError, match="re-fetch it with --overwrite"):
        _run_command(_make_app(), tmp_path, True, False, [], False)


def test_run_run_rejects_id_and_path(tmp_path):
    with pytest.raises(QAIHubAppsError, match="Cannot specify both"):
        run_run("test_app", tmp_path, tmp_path, MagicMock(), None)


def test_run_run_from_path_warns_about_model(
    tmp_path, stub_run_run, monkeypatch, caplog
):
    monkeypatch.setattr(
        run_mod, "_resolve_app_from_dir", MagicMock(return_value=_make_app())
    )
    monkeypatch.setattr(run_mod, "resolve_device_info", MagicMock(return_value=DEVICE))
    run_run(None, tmp_path, tmp_path, MagicMock(), ModelAsset(model_id="m"))
    assert "--model/--model-id are not" in caplog.text
    stub_run_run.run_build.assert_not_called()
    assert stub_run_run.subprocess_run.call_args.kwargs["cwd"] == tmp_path.resolve()


def test_run_run_windows_app_runs_natively(tmp_path, stub_run_run, caplog):
    registry = MagicMock()
    registry.find_by_id.return_value = _make_app(AppType.WINDOWS)
    run_run("test_app", None, tmp_path, registry, None, use_docker=True)
    assert "run natively" in caplog.text
    # The build still honors --docker; only the run is forced native.
    assert stub_run_run.run_build.call_args.kwargs["use_docker"] is True
    assert stub_run_run.run_command.call_args.args[2] is False
    # No model given: the app's first related model is used for the run device.
    assert stub_run_run.run_build.call_args.args[4] == ModelAsset(
        model_id="test_model", device=DEVICE.name
    )


def test_run_run_device_override(tmp_path, stub_run_run, monkeypatch):
    override = make_device(name="Device B")
    monkeypatch.setattr(
        run_mod, "resolve_device_info", MagicMock(return_value=override)
    )
    registry = MagicMock()
    registry.find_by_id.return_value = _make_app()
    run_run(
        "test_app",
        None,
        tmp_path,
        registry,
        ModelAsset(model_id="m", device="Device B"),
    )
    assert stub_run_run.ensure_run_supported.call_args.args[1] is override


def test_run_run_android_prompts_for_device(tmp_path, stub_run_run, monkeypatch):
    android = [make_device(name="Device A"), make_device(name="Device B")]
    monkeypatch.setattr(
        run_mod, "list_android_devices", MagicMock(return_value=android)
    )
    prompt = MagicMock(return_value=android[1])
    monkeypatch.setattr(run_mod, "prompt_for_device", prompt)
    registry = MagicMock()
    registry.find_by_id.return_value = _make_app(
        AppType.ANDROID, supported_devices=["Device B"]
    )
    run_run("test_app", None, tmp_path, registry, None)
    # Only the app's supported devices are offered.
    assert prompt.call_args.args[0] == [android[1]]
    assert stub_run_run.ensure_run_supported.call_args.args[1] is android[1]


def test_run_run_raises_when_device_still_unset(tmp_path, stub_run_run, monkeypatch):
    monkeypatch.setattr(run_mod, "get_configured_device", MagicMock(return_value=None))
    monkeypatch.setattr(run_mod, "run_configure", MagicMock())
    registry = MagicMock()
    registry.find_by_id.return_value = _make_app()
    with pytest.raises(QAIHubAppsError, match="No target device configured"):
        run_run("test_app", None, tmp_path, registry, None)


def test_run_run_subprocess_failure_raises(tmp_path, stub_run_run, monkeypatch):
    monkeypatch.setattr(
        run_mod.subprocess,
        "run",
        MagicMock(side_effect=subprocess.CalledProcessError(3, "bash")),
    )
    registry = MagicMock()
    registry.find_by_id.return_value = _make_app()
    with pytest.raises(QAIHubAppsError, match="exit code 3"):
        run_run("test_app", None, tmp_path, registry, None)
