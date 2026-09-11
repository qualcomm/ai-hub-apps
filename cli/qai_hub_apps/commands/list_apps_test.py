# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qai_hub_apps.commands.list_apps import run_info, run_list
from qai_hub_apps.configs.app_yaml import AppLanguage, AppType
from qai_hub_apps.conftest import make_app_info
from qai_hub_apps.errors import AppNotFoundError
from qai_hub_apps.registry.base import App
from qai_hub_apps.registry.filters import AppFilter


def _make_registry_with_apps(*infos):
    registry = MagicMock()
    registry.apps = [App(info) for info in infos]
    registry.filter.side_effect = lambda f: [a for a in registry.apps if f.matches(a)]
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


def _two_app_registry():
    return _make_registry_with_apps(
        make_app_info(id="app_ubuntu", app_type=AppType.UBUNTU),
        make_app_info(id="app_android", app_type=AppType.ANDROID),
    )


def test_run_list_filter_narrows_rows(capsys):
    run_list(_two_app_registry(), AppFilter(app_types=frozenset({AppType.ANDROID})))
    out = capsys.readouterr().out
    assert "app_android" in out
    assert "app_ubuntu" not in out


def test_run_list_filtered_total_shows_denominator(capsys):
    run_list(_two_app_registry(), AppFilter(app_types=frozenset({AppType.ANDROID})))
    assert "Total: 1 of 2 apps" in capsys.readouterr().out


def test_run_list_empty_filter_keeps_plain_total(capsys):
    run_list(_two_app_registry(), AppFilter())
    out = capsys.readouterr().out
    assert "Total: 2 apps" in out
    assert "No apps match" not in out


def test_run_list_no_matches_message(capsys):
    run_list(_two_app_registry(), AppFilter(models=("nope",)))
    out = capsys.readouterr().out
    assert "No apps match those filters." in out
    assert "See all apps:" in out
    assert "qai-hub-apps list" in out
    assert "Total:" not in out


def test_run_list_empty_registry_without_filter_prints_table(capsys):
    run_list(_make_registry_with_apps())
    out = capsys.readouterr().out
    assert "0 apps" in out
    assert "No apps match" not in out


@pytest.mark.parametrize(
    ("app_filter", "expected"),
    [
        (AppFilter(runtimes=frozenset({"nope"})), "Available runtimes: onnx."),
        (AppFilter(domains=frozenset({"nope"})), "Available domains: Test."),
        (AppFilter(use_cases=frozenset({"nope"})), "Available use cases: Testing."),
        (AppFilter(precisions=frozenset({"nope"})), "Available precisions: float."),
    ],
)
def test_run_list_no_matches_lists_available_values(app_filter, expected, capsys):
    run_list(_two_app_registry(), app_filter)
    assert expected in capsys.readouterr().out


def test_run_list_no_matches_omits_hints_for_model_and_device(capsys):
    run_list(_two_app_registry(), AppFilter(models=("nope",), device="Nope"))
    assert "Available" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("app_filter", "header", "cell"),
    [
        (AppFilter(app_types=frozenset({AppType.UBUNTU})), "Type", "ubuntu"),
        (AppFilter(runtimes=frozenset({"onnx"})), "Runtime", "onnx"),
        (AppFilter(use_cases=frozenset({"testing"})), "Use Case", "Testing"),
        (AppFilter(precisions=frozenset({"float"})), "Precision", "float"),
        (AppFilter(models=("test",)), "Models", "test_model"),
        (
            AppFilter(device="Snapdragon 8 Elite QRD"),
            "Supported Devices",
            "Snapdragon 8 Elite QRD",
        ),
    ],
)
def test_filter_adds_its_column(app_filter, header, cell, capsys):
    registry = _make_registry_with_apps(
        make_app_info(supported_devices=["Snapdragon 8 Elite QRD"])
    )
    run_list(registry, app_filter)
    out = capsys.readouterr().out
    assert header in out
    assert cell in out


@pytest.mark.parametrize(
    ("app_filter", "column"),
    [
        (AppFilter(languages=frozenset({AppLanguage.PYTHON})), "Languages"),
        (AppFilter(domains=frozenset({"test"})), "Domain"),
    ],
)
def test_filter_on_base_column_does_not_duplicate_it(app_filter, column, capsys):
    run_list(_make_registry_with_apps(make_app_info()), app_filter)
    header = capsys.readouterr().out.splitlines()[3]
    assert header.count(column) == 1


def test_column_order_is_independent_of_filter_order(capsys):
    app_filter = AppFilter(
        models=("test",),
        app_types=frozenset({AppType.UBUNTU}),
        device="Snapdragon 8 Elite QRD",
    )
    run_list(_make_registry_with_apps(make_app_info()), app_filter)
    out = capsys.readouterr().out
    assert out.index("Type") < out.index("Models") < out.index("Supported Devices")


@pytest.mark.parametrize(
    ("app_filter", "expected"),
    [
        (AppFilter(), ("Name", False)),
        (
            AppFilter(models=("test",), device="Nope", runtimes=frozenset({"onnx"})),
            ("Models", True),
        ),
        (
            AppFilter(device="Nope", runtimes=frozenset({"onnx"})),
            ("Supported Devices", True),
        ),
        (AppFilter(runtimes=frozenset({"onnx"})), ("Runtime", True)),
    ],
)
def test_wrap_target(app_filter, expected, monkeypatch):
    mock_build_table = MagicMock(return_value="")
    monkeypatch.setattr("qai_hub_apps.commands.list_apps.build_table", mock_build_table)
    run_list(_make_registry_with_apps(make_app_info()), app_filter)
    kwargs = mock_build_table.call_args.kwargs
    assert (kwargs["wrap_column"], kwargs["wrap_on_commas"]) == expected
