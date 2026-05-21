# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qai_hub_apps.configs.app_yaml import AppLanguage, AppUrl
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.conftest import make_app_info
from qai_hub_apps.errors import (
    AppIncompatibleError,
    AppNotFoundError,
    ModelAssetNotFoundError,
    QAIHubAppsError,
)
from qai_hub_apps.registry.base import App, Registry, _make_app
from qai_hub_apps.registry.python_app import PythonApp


def fake_download(url: str, path: Path, extract: bool = False) -> Path:
    """Simulate download: for model URLs creates metadata.json + model files + an extra file."""
    import json

    path.mkdir(parents=True, exist_ok=True)
    if "model" in url:
        metadata: dict[str, dict[str, dict]] = {
            "model_files": {"model1.onnx": {}, "model2.onnx": {}}
        }
        (path / "metadata.json").write_text(json.dumps(metadata))
        (path / "model1.onnx").touch()
        (path / "model2.onnx").touch()
        (path / "LICENSE").touch()
    return path


def test_make_app_returns_python_app_for_python_language():
    info = make_app_info(languages=[AppLanguage.PYTHON])
    app = _make_app(info)
    assert isinstance(app, PythonApp)


def test_make_app_returns_base_app_for_non_python():
    info = make_app_info(languages=[AppLanguage.CPP])
    app = _make_app(info)
    assert type(app) is App


def test_make_app_returns_base_app_for_empty_languages():
    info = make_app_info(languages=[])
    app = _make_app(info)
    assert type(app) is App


def test_registry_version_returns_dev_when_none(tmp_path, monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: True)
    content = """\
schema_version: '1.1'
min_cli_version: 0.0.1
apps: []
"""
    p = tmp_path / "registry.yaml"
    p.write_text(content)
    registry = Registry.load(p)
    assert registry.version == "dev"


def test_registry_version_returns_string_when_set(tmp_path, monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: True)
    content = """\
schema_version: '1.1'
min_cli_version: 0.0.1
version: '1.2.3'
apps: []
"""
    p = tmp_path / "registry.yaml"
    p.write_text(content)
    registry = Registry.load(p)
    assert registry.version == "1.2.3"


def test_find_by_id_exact_match(sample_registry_yaml):
    registry = Registry.load(sample_registry_yaml)
    app = registry.find_by_id("test_app")
    assert app.id == "test_app"


def test_find_by_id_case_insensitive(sample_registry_yaml):
    registry = Registry.load(sample_registry_yaml)
    app = registry.find_by_id("TEST_APP")
    assert app.id == "test_app"


def test_find_by_id_raises_app_not_found(sample_registry_yaml):
    registry = Registry.load(sample_registry_yaml)
    with pytest.raises(AppNotFoundError):
        registry.find_by_id("nonexistent_app")


def test_registry_load_singleton(sample_registry_yaml):
    r1 = Registry.load(sample_registry_yaml)
    r2 = Registry.load(sample_registry_yaml)
    assert r1 is r2


def test_registry_load_fresh_after_reset(sample_registry_yaml):
    r1 = Registry.load(sample_registry_yaml)
    Registry._instance = None
    r2 = Registry.load(sample_registry_yaml)
    assert r1 is not r2


def test_fetch_with_url_calls_download(monkeypatch, tmp_path):
    dest = tmp_path / "output"
    dest.mkdir()
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)

    info = make_app_info(url=AppUrl(source="https://example.com/app.zip"))
    app = App(info)
    result = app.fetch(dest)

    assert result == dest / "test_app"
    assert result.exists()


def test_fetch_with_url_dev_also_uses_download(monkeypatch, tmp_path):
    """URL present in dev mode → still downloads normally."""
    dest = tmp_path / "output"
    dest.mkdir()
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: True)

    info = make_app_info(url=AppUrl(source="https://example.com/app.zip"))
    app = App(info)
    result = app.fetch(dest)
    assert result == dest / "test_app"
    assert result.exists()


def test_fetch_dev_no_url_calls_bundle_app(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: True)

    def fake_bundle_app(app_id: str, dest: Path, make_zip: bool = True) -> None:
        (dest / app_id).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("qai_hub_apps.registry.base._bundle_app", fake_bundle_app)

    mock_download = MagicMock()
    monkeypatch.setattr("qai_hub_apps.registry.base.download", mock_download)

    info = make_app_info(url=None)
    app = App(info)
    result = app.fetch(tmp_path)

    assert result == tmp_path / "test_app"
    assert result.exists()
    mock_download.assert_not_called()


def test_fetch_dev_no_url_missing_bundler_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: True)
    monkeypatch.setattr("qai_hub_apps.registry.base._bundle_app", None)

    info = make_app_info(url=None)
    app = App(info)
    with pytest.raises(QAIHubAppsError, match="qai_hub_apps_test"):
        app.fetch(tmp_path)


def test_fetch_prod_no_url_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    info = make_app_info(url=None)
    app = App(info)
    with pytest.raises(QAIHubAppsError):
        app.fetch(tmp_path)


def test_fetch_model_not_in_related_models_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["valid_model"],
        model_file_paths=["models/model.onnx"],
    )
    app = App(info)
    asset = ModelAsset(model_id="wrong_model", chipset=None)
    with pytest.raises(AppIncompatibleError, match="wrong_model"):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_no_model_location_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=[],
        model_file_dir=None,
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    with pytest.raises(
        AppIncompatibleError, match="model_file_paths or model_file_dir"
    ):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_model_file_dir_success(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=[],
        model_file_dir="models",
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    result = app.fetch(tmp_path, model_asset=asset)

    assert result == tmp_path / "test_app"
    assert (result / "models" / "model1.onnx").exists()
    assert (result / "models" / "model2.onnx").exists()
    assert (result / "models" / "LICENSE").exists()
    assert (result / "models" / "metadata.json").exists()


def test_fetch_model_asset_not_found_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(side_effect=FileNotFoundError("not found")),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/model.onnx"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    with pytest.raises(ModelAssetNotFoundError):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_model_asset_success(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/renamed_model1.onnx", "models/renamed_model2.onnx"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    result = app.fetch(tmp_path, model_asset=asset)

    import json

    assert result == tmp_path / "test_app"
    assert result.exists()
    assert (result / "models" / "renamed_model1.onnx").exists()
    assert (result / "models" / "renamed_model2.onnx").exists()
    assert (result / "models" / "LICENSE").exists()
    metadata = json.loads((result / "models" / "metadata.json").read_text())
    assert list(metadata["model_files"].keys()) == [
        "renamed_model1.onnx",
        "renamed_model2.onnx",
    ]


def test_fetch_model_failure_leaves_no_dest(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )

    def fail_model_download(url: str, path: Path, extract: bool = False) -> Path:
        if "model" in url:
            raise RuntimeError("model download failed")
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr("qai_hub_apps.registry.base.download", fail_model_download)

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/model.onnx"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    with pytest.raises(RuntimeError):
        app.fetch(tmp_path, model_asset=asset)

    assert not (tmp_path / "test_app").exists()


def test_fetch_model_asset_not_found_leaves_no_dest(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(side_effect=FileNotFoundError("not found")),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/model.onnx"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    with pytest.raises(ModelAssetNotFoundError):
        app.fetch(tmp_path, model_asset=asset)

    assert not (tmp_path / "test_app").exists()


def test_fetch_model_file_paths_renames_using_metadata(monkeypatch, tmp_path):
    """Files from metadata.json are placed at model_file_paths destinations (with rename)."""
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=[
            "models/PalmDetector.tflite",
            "models/HandLandmarkDetector.tflite",
        ],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    result = app.fetch(tmp_path, model_asset=asset)

    assert (result / "models" / "PalmDetector.tflite").exists()
    assert (result / "models" / "HandLandmarkDetector.tflite").exists()


def test_fetch_model_file_paths_count_mismatch_raises(monkeypatch, tmp_path):
    """AppIncompatibleError when metadata.json count differs from model_file_paths count."""
    import json

    def fake_download_one_file(url: str, path: Path, extract: bool = False) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        if "model" in url:
            metadata: dict[str, dict[str, dict]] = {"model_files": {"model.onnx": {}}}
            (path / "metadata.json").write_text(json.dumps(metadata))
            (path / "model.onnx").touch()
        return path

    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download_one_file)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/a.onnx", "models/b.onnx"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    with pytest.raises(AppIncompatibleError, match="1 file\\(s\\) but 2 were expected"):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_model_file_paths_different_dirs_raises(monkeypatch, tmp_path):
    """AppIncompatibleError when model_file_paths span different parent directories."""
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/a.onnx", "assets/b.onnx"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    with pytest.raises(AppIncompatibleError, match="same parent directory"):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_model_missing_metadata_json_raises(monkeypatch, tmp_path):
    """A model asset without metadata.json raises AppIncompatibleError."""

    def fake_download_no_metadata(url: str, path: Path, extract: bool = False) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        if "model" in url:
            (path / "model.onnx").touch()
        return path

    monkeypatch.setattr(
        "qai_hub_apps.registry.base.download", fake_download_no_metadata
    )
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/model.onnx"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    with pytest.raises(AppIncompatibleError, match="is missing metadata\.json"):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_dest_exists_uses_next_free_path(monkeypatch, tmp_path):
    (tmp_path / "test_app").mkdir()
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)

    info = make_app_info(url=AppUrl(source="https://example.com/app.zip"))
    app = App(info)
    result = app.fetch(tmp_path)

    assert result == tmp_path / "test_app-1"
    assert result.exists()


def test_detail_fields_contains_id_and_type():
    app = App(make_app_info(id="my_app"))
    fields = dict(app.detail_fields())
    assert "ID" in fields
    assert fields["ID"] == "my_app"
    assert "Type" in fields


def test_detail_fields_includes_runtime_when_set():
    app = App(make_app_info())
    fields = dict(app.detail_fields())
    assert "Runtime" in fields


def test_detail_fields_skips_empty_domain():
    app = App(make_app_info(domain=""))
    fields = dict(app.detail_fields())
    assert "Domain" not in fields


def test_registry_apps_returns_all(sample_registry_yaml):
    registry = Registry.load(sample_registry_yaml)
    apps = list(registry.apps)
    assert len(apps) == 1
    assert apps[0].id == "test_app"


def test_fetch_app_unsupported_platform_warns(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.is_app_supported", lambda app: False
    )

    from qai_hub_apps.configs.app_yaml import AppUrl
    from qai_hub_apps.configs.registry_yaml import AppRegistry

    info = make_app_info(url=AppUrl(source="https://example.com/app.zip"))
    raw = AppRegistry(schema_version="1.1", min_cli_version="0.0.1", apps=[info])
    registry = Registry(raw)
    registry.fetch_app("test_app", tmp_path)

    assert (
        "Warning: This app may not be supported on the current device."
        in capsys.readouterr().out
    )
