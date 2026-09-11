# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qai_hub_apps.configs.app_yaml import AppLanguage, AppType
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.conftest import make_device
from qai_hub_apps.errors import QAIHubAppsError
from qai_hub_apps.main import main
from qai_hub_apps.registry import AppFilter


def _run_main(argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["qai-hub-apps", *argv])
    main()


def test_list_command_calls_run_list(monkeypatch, sample_registry_yaml):
    mock_run_list = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_list", mock_run_list)
    _run_main(["list", "--registry", str(sample_registry_yaml)], monkeypatch)
    mock_run_list.assert_called_once()


def test_info_command_calls_run_info(monkeypatch, sample_registry_yaml):
    mock_run_info = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_info", mock_run_info)
    _run_main(
        ["info", "test_app", "--registry", str(sample_registry_yaml)], monkeypatch
    )
    mock_run_info.assert_called_once()
    call_app_id = mock_run_info.call_args[0][0]
    assert call_app_id == "test_app"


def test_fetch_command_calls_run_fetch(monkeypatch, tmp_path, sample_registry_yaml):
    mock_run_fetch = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_fetch", mock_run_fetch)
    _run_main(
        [
            "fetch",
            "test_app",
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(sample_registry_yaml),
        ],
        monkeypatch,
    )
    mock_run_fetch.assert_called_once()
    call_app_id, call_dest = mock_run_fetch.call_args[0][:2]
    assert call_app_id == "test_app"
    assert call_dest == tmp_path


def test_fetch_command_with_model_creates_model_asset(
    monkeypatch, tmp_path, sample_registry_yaml
):
    mock_run_fetch = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_fetch", mock_run_fetch)
    _run_main(
        [
            "fetch",
            "test_app",
            "--model",
            "whisper_base",
            "--chipset",
            "snapdragon_8_gen_3",
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(sample_registry_yaml),
        ],
        monkeypatch,
    )
    mock_run_fetch.assert_called_once()
    # run_fetch is called positionally: (app_id, dest, registry, model_asset)
    model_asset = mock_run_fetch.call_args[0][3]
    assert isinstance(model_asset, ModelAsset)
    assert model_asset.model_id == "whisper_base"
    assert model_asset.chipset == "snapdragon_8_gen_3"


def test_fetch_command_with_model_path_creates_local_asset(
    monkeypatch, tmp_path, sample_registry_yaml
):
    export_dir = tmp_path / "exported"
    export_dir.mkdir()

    mock_run_fetch = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_fetch", mock_run_fetch)
    _run_main(
        [
            "fetch",
            "test_app",
            "--model",
            str(export_dir),
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(sample_registry_yaml),
        ],
        monkeypatch,
    )
    model_asset = mock_run_fetch.call_args[0][3]
    assert isinstance(model_asset, ModelAsset)
    assert model_asset.path == export_dir
    assert model_asset.model_id is None


def test_fetch_command_with_model_id_flag(monkeypatch, tmp_path, sample_registry_yaml):
    mock_run_fetch = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_fetch", mock_run_fetch)
    _run_main(
        [
            "fetch",
            "test_app",
            "--model-id",
            "whisper_base",
            "--chipset",
            "snapdragon_8_gen_3",
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(sample_registry_yaml),
        ],
        monkeypatch,
    )
    model_asset = mock_run_fetch.call_args[0][3]
    assert isinstance(model_asset, ModelAsset)
    assert model_asset.model_id == "whisper_base"
    assert model_asset.chipset == "snapdragon_8_gen_3"
    assert model_asset.path is None


def test_fetch_command_with_model_path_flag_resolves_absolute(
    monkeypatch, tmp_path, sample_registry_yaml
):
    export_dir = tmp_path / "exported"
    export_dir.mkdir()

    mock_run_fetch = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_fetch", mock_run_fetch)
    _run_main(
        [
            "fetch",
            "test_app",
            "--model-path",
            str(export_dir),
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(sample_registry_yaml),
        ],
        monkeypatch,
    )
    model_asset = mock_run_fetch.call_args[0][3]
    assert isinstance(model_asset, ModelAsset)
    assert model_asset.path == export_dir.resolve()
    assert model_asset.model_id is None


def test_fetch_chipset_with_model_path_exits(
    monkeypatch, tmp_path, sample_registry_yaml
):
    monkeypatch.setattr("qai_hub_apps.main.run_fetch", MagicMock())
    with pytest.raises(SystemExit):
        _run_main(
            [
                "fetch",
                "test_app",
                "--model-path",
                str(tmp_path),
                "--chipset",
                "snapdragon_8_gen_3",
                "--registry",
                str(sample_registry_yaml),
            ],
            monkeypatch,
        )


def test_fetch_model_path_with_chipset_warns(
    monkeypatch, tmp_path, sample_registry_yaml, caplog
):
    export_dir = tmp_path / "exported"
    export_dir.mkdir()

    monkeypatch.setattr("qai_hub_apps.main.run_fetch", MagicMock())
    _run_main(
        [
            "fetch",
            "test_app",
            "--model",
            str(export_dir),
            "--chipset",
            "snapdragon_8_gen_3",
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(sample_registry_yaml),
        ],
        monkeypatch,
    )
    assert "--chipset is ignored" in caplog.text


def test_fetch_without_model_passes_none(monkeypatch, tmp_path, sample_registry_yaml):
    mock_run_fetch = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_fetch", mock_run_fetch)
    _run_main(
        [
            "fetch",
            "test_app",
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(sample_registry_yaml),
        ],
        monkeypatch,
    )
    # run_fetch is called positionally: (app_id, dest, registry, model_asset)
    model_asset = mock_run_fetch.call_args[0][3]
    assert model_asset is None


def test_chipset_without_model_exits(monkeypatch, tmp_path, sample_registry_yaml):
    with pytest.raises(SystemExit) as exc:
        _run_main(
            [
                "fetch",
                "test_app",
                "--chipset",
                "snapdragon_8_gen_3",
                "--output-dir",
                str(tmp_path),
                "--registry",
                str(sample_registry_yaml),
            ],
            monkeypatch,
        )
    assert exc.value.code == 2


def test_fetch_device_and_chipset_mutually_exclusive(
    monkeypatch, tmp_path, sample_registry_yaml
):
    with pytest.raises(SystemExit) as exc:
        _run_main(
            [
                "fetch",
                "test_app",
                "--model-id",
                "whisper_base",
                "--chipset",
                "some-chipset",
                "--device",
                "Some Device",
                "--registry",
                str(sample_registry_yaml),
            ],
            monkeypatch,
        )
    assert exc.value.code == 2


def test_fetch_device_without_model_exits(monkeypatch, tmp_path, sample_registry_yaml):
    with pytest.raises(SystemExit) as exc:
        _run_main(
            [
                "fetch",
                "test_app",
                "--device",
                "Some Device",
                "--output-dir",
                str(tmp_path),
                "--registry",
                str(sample_registry_yaml),
            ],
            monkeypatch,
        )
    assert exc.value.code == 2


def test_missing_registry_exits_1(monkeypatch, tmp_path):
    nonexistent = tmp_path / "nonexistent.yaml"
    with pytest.raises(SystemExit) as exc:
        _run_main(["list", "--registry", str(nonexistent)], monkeypatch)
    assert exc.value.code == 1


def test_missing_registry_prints_message(monkeypatch, tmp_path, caplog):
    nonexistent = tmp_path / "nonexistent.yaml"
    with pytest.raises(SystemExit):
        _run_main(["list", "--registry", str(nonexistent)], monkeypatch)
    assert "Registry not found" in caplog.text


def test_qai_hub_apps_error_exits_1(monkeypatch, sample_registry_yaml):
    monkeypatch.setattr(
        "qai_hub_apps.main.run_list",
        MagicMock(side_effect=QAIHubAppsError("something went wrong")),
    )
    with pytest.raises(SystemExit) as exc:
        _run_main(["list", "--registry", str(sample_registry_yaml)], monkeypatch)
    assert exc.value.code == 1


def test_qai_hub_apps_error_prints_message(monkeypatch, sample_registry_yaml, caplog):
    monkeypatch.setattr(
        "qai_hub_apps.main.run_list",
        MagicMock(side_effect=QAIHubAppsError("something went wrong")),
    )
    with pytest.raises(SystemExit):
        _run_main(["list", "--registry", str(sample_registry_yaml)], monkeypatch)
    assert "something went wrong" in caplog.text


def test_no_command_does_not_crash(monkeypatch, capsys):
    # No subcommand → prints help, exits 0 (--registry is per-subparser, not global)
    _run_main([], monkeypatch)
    out = capsys.readouterr().out
    assert "usage" in out.lower() or "qai-hub-apps" in out


def _run_build_main(argv, monkeypatch, sample_registry_yaml):
    """Run `build ...` with the experimental gate open and run_build mocked."""
    monkeypatch.setenv("QAI_HUB_APPS_EXPERIMENTAL", "1")
    mock_run_build = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_build", mock_run_build)
    _run_main(["build", *argv, "--registry", str(sample_registry_yaml)], monkeypatch)
    return mock_run_build


def test_build_positional_id(monkeypatch, sample_registry_yaml):
    mock = _run_build_main(["test_app"], monkeypatch, sample_registry_yaml)
    mock.assert_called_once()
    app_id, app_path = mock.call_args.args[0], mock.call_args.args[1]
    assert app_id == "test_app"
    assert app_path is None


def test_build_positional_path(monkeypatch, tmp_path, sample_registry_yaml):
    app_dir = tmp_path / "test_app"
    app_dir.mkdir()
    mock = _run_build_main([str(app_dir)], monkeypatch, sample_registry_yaml)
    app_id, app_path = mock.call_args.args[0], mock.call_args.args[1]
    assert app_id is None
    assert app_path == app_dir


def test_build_explicit_app_id(monkeypatch, sample_registry_yaml):
    mock = _run_build_main(["--app-id", "test_app"], monkeypatch, sample_registry_yaml)
    assert mock.call_args.args[0] == "test_app"


def test_build_explicit_app_path(monkeypatch, tmp_path, sample_registry_yaml):
    mock = _run_build_main(
        ["--app-path", str(tmp_path)], monkeypatch, sample_registry_yaml
    )
    assert mock.call_args.args[1] == tmp_path


def test_build_no_docker_and_clean(monkeypatch, sample_registry_yaml):
    mock = _run_build_main(
        ["test_app", "--no-docker", "--clean"], monkeypatch, sample_registry_yaml
    )
    assert mock.call_args.kwargs["use_docker"] is False
    assert mock.call_args.kwargs["clean"] is True


def test_build_positional_path_with_overwrite_errors(
    monkeypatch, tmp_path, sample_registry_yaml
):
    app_dir = tmp_path / "test_app"
    app_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        _run_build_main(
            [str(app_dir), "--overwrite"], monkeypatch, sample_registry_yaml
        )
    assert exc.value.code == 1


def test_build_requires_a_target(monkeypatch, sample_registry_yaml):
    with pytest.raises(SystemExit) as exc:
        _run_build_main([], monkeypatch, sample_registry_yaml)
    assert exc.value.code == 2


def test_default_registry_calls_ensure_registry(monkeypatch, sample_registry_yaml):
    mock_ensure = MagicMock(return_value=sample_registry_yaml)
    monkeypatch.setattr("qai_hub_apps.registry.base.ensure_registry", mock_ensure)
    mock_run_list = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_list", mock_run_list)

    _run_main(["list"], monkeypatch)

    mock_ensure.assert_called_once()


def _run_experimental_main(argv: list[str], monkeypatch, target: str) -> MagicMock:
    """Run `argv` with the experimental gate open and `target` in main mocked."""
    monkeypatch.setenv("QAI_HUB_APPS_EXPERIMENTAL", "1")
    mock = MagicMock()
    monkeypatch.setattr(f"qai_hub_apps.main.{target}", mock)
    _run_main(argv, monkeypatch)
    return mock


def _run_run_main(argv: list[str], monkeypatch, sample_registry_yaml) -> MagicMock:
    """Run a `run`/`test` invocation against the sample registry with run_run mocked."""
    command, *rest = argv
    return _run_experimental_main(
        [command, "--registry", str(sample_registry_yaml), *rest],
        monkeypatch,
        "run_run",
    )


def test_run_command_passes_through_app_args(monkeypatch, sample_registry_yaml):
    mock = _run_run_main(
        ["run", "test_app", "--", "--verbose", "1"], monkeypatch, sample_registry_yaml
    )
    assert mock.call_args.kwargs["app_args"] == ["--verbose", "1"]


def test_run_command_rejects_chipset(monkeypatch, sample_registry_yaml):
    with pytest.raises(SystemExit) as exc:
        _run_run_main(
            [
                "run",
                "test_app",
                "--model",
                "whisper_base",
                "--chipset",
                "snapdragon_8_gen_3",
            ],
            monkeypatch,
            sample_registry_yaml,
        )
    assert exc.value.code == 2


def test_configure_show_calls_run_configure(monkeypatch):
    mock = _run_experimental_main(["configure", "--show"], monkeypatch, "run_configure")
    mock.assert_called_once_with(None, show=True)


def _run_list_main(argv: list[str], monkeypatch, sample_registry_yaml) -> AppFilter:
    """Run `list ...`; return the AppFilter run_list received."""
    mock = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_list", mock)
    _run_main(["list", "--registry", str(sample_registry_yaml), *argv], monkeypatch)
    return mock.call_args[0][1]


def _run_list_device_main(
    argv: list[str], monkeypatch, sample_registry_yaml
) -> AppFilter:
    """Run `list --device ...` with the experimental gate open."""
    monkeypatch.setenv("QAI_HUB_APPS_EXPERIMENTAL", "1")
    return _run_list_main(argv, monkeypatch, sample_registry_yaml)


def test_list_without_filters_passes_empty_filter(monkeypatch, sample_registry_yaml):
    mock_run_list = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.run_list", mock_run_list)
    _run_main(["list", "--registry", str(sample_registry_yaml)], monkeypatch)
    assert mock_run_list.call_args[0][1].is_empty()


def test_list_device_filter_requires_experimental(monkeypatch, sample_registry_yaml):
    with pytest.raises(SystemExit) as exc:
        _run_main(
            ["list", "--device", "Device A", "--registry", str(sample_registry_yaml)],
            monkeypatch,
        )
    assert exc.value.code == 2


def test_list_rejects_a_positional_argument(monkeypatch, sample_registry_yaml):
    with pytest.raises(SystemExit) as exc:
        _run_main(
            ["list", "whisper", "--registry", str(sample_registry_yaml)], monkeypatch
        )
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--type", "android", "windows"],
        ["--type", "android", "--type", "windows"],
    ],
)
def test_list_type_accepts_multiple_values(argv, monkeypatch, sample_registry_yaml):
    app_filter = _run_list_main(argv, monkeypatch, sample_registry_yaml)
    assert app_filter.app_types == frozenset({AppType.ANDROID, AppType.WINDOWS})


def test_list_all_filter_flags(monkeypatch, sample_registry_yaml):
    app_filter = _run_list_main(
        [
            "--language",
            "Python",
            "--runtime",
            "onnx",
            "--domain",
            "Audio",
            "--use-case",
            "Speech Recognition",
            "--precision",
            "float",
            "--model",
            "whisper",
        ],
        monkeypatch,
        sample_registry_yaml,
    )
    assert app_filter.languages == frozenset({AppLanguage.PYTHON})
    assert app_filter.runtimes == frozenset({"onnx"})
    assert app_filter.domains == frozenset({"audio"})
    assert app_filter.use_cases == frozenset({"speech recognition"})
    assert app_filter.precisions == frozenset({"float"})
    assert app_filter.models == ("whisper",)


def test_list_invalid_type_exits_1(monkeypatch, sample_registry_yaml, caplog):
    with pytest.raises(SystemExit) as exc:
        _run_main(
            ["list", "--type", "macos", "--registry", str(sample_registry_yaml)],
            monkeypatch,
        )
    assert exc.value.code == 1
    assert "Valid app types" in caplog.text


def test_list_device_with_value_is_resolved(monkeypatch, sample_registry_yaml):
    monkeypatch.setattr(
        "qai_hub_apps.main.resolve_device_info",
        MagicMock(return_value=make_device(name="Device A")),
    )
    app_filter = _run_list_device_main(
        ["--device", "device a"], monkeypatch, sample_registry_yaml
    )
    assert app_filter.device == "Device A"


def test_list_device_without_value_uses_configured_device(
    monkeypatch, sample_registry_yaml
):
    monkeypatch.setattr(
        "qai_hub_apps.main.get_configured_device",
        MagicMock(return_value=make_device(name="Device A")),
    )
    mock_resolve = MagicMock()
    monkeypatch.setattr("qai_hub_apps.main.resolve_device_info", mock_resolve)
    app_filter = _run_list_device_main(["--device"], monkeypatch, sample_registry_yaml)
    assert app_filter.device == "Device A"
    mock_resolve.assert_not_called()


def test_list_device_without_value_and_none_configured_exits_1(
    monkeypatch, sample_registry_yaml, caplog
):
    monkeypatch.setenv("QAI_HUB_APPS_EXPERIMENTAL", "1")
    monkeypatch.setattr(
        "qai_hub_apps.main.get_configured_device", MagicMock(return_value=None)
    )
    with pytest.raises(SystemExit) as exc:
        _run_main(
            ["list", "--device", "--registry", str(sample_registry_yaml)], monkeypatch
        )
    assert exc.value.code == 1
    assert "No target device configured" in caplog.text
