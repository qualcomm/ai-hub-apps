# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qai_hub_models_cli.proto_helpers.release_assets import AssetNotFoundError

from qai_hub_apps.configs.app_yaml import AppLanguage, AppUrl
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.conftest import make_app_info
from qai_hub_apps.errors import (
    AppIncompatibleError,
    AppNotFoundError,
    InvalidArgumentError,
    ModelAssetNotFoundError,
    QAIHubAppsError,
)
from qai_hub_apps.registry.base import (
    DEFAULT_DEPRECATION_MESSAGE,
    App,
    Registry,
    _make_app,
)
from qai_hub_apps.registry.python_app import PythonApp


def _make_export_dir(
    parent: Path,
    *,
    model_files: list[str],
    model_id: str | None = "test_model",
    with_metadata: bool = True,
    as_zip: bool = False,
) -> Path:
    """Create a locally-exported model (metadata.json + model files).

    Returns the export directory, or — when ``as_zip`` is True — a ``.zip`` of it.
    """
    import json
    import zipfile

    export = parent / "export"
    export.mkdir()
    for name in model_files:
        (export / name).touch()
    if with_metadata:
        metadata: dict = {"model_files": {name: {} for name in model_files}}
        if model_id is not None:
            metadata["model_id"] = model_id
        (export / "metadata.json").write_text(json.dumps(metadata))
    if as_zip:
        zip_path = parent / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for item in export.iterdir():
                zf.write(item, item.name)
        return zip_path
    return export


def _make_fake_download(
    *,
    model_files: list[str] | None = None,
    extra_files: list[str] | None = None,
    model_id: str | None = "test_model",
    with_metadata: bool = True,
):
    """Build a fake `download` for model assets with customizable contents.

    For model URLs, writes the given model files (and `extra_files`) plus a
    metadata.json describing them. Set `with_metadata=False` to omit it, or
    `model_id=None` to leave that field out of the metadata.
    """
    model_files = ["model1.onnx", "model2.onnx"] if model_files is None else model_files
    extra_files = ["LICENSE"] if extra_files is None else extra_files

    def fake_download(
        url: str, path: Path, extract: bool = False, quiet: bool = False
    ) -> Path:
        import json

        path.mkdir(parents=True, exist_ok=True)
        if "model" in url:
            for name in (*model_files, *extra_files):
                (path / name).touch()
            if with_metadata:
                metadata: dict = {"model_files": {name: {} for name in model_files}}
                if model_id is not None:
                    metadata["model_id"] = model_id
                (path / "metadata.json").write_text(json.dumps(metadata))
        return path

    return fake_download


# Default fake download: two model files + LICENSE, with valid metadata.
fake_download = _make_fake_download()


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


def test_fetch_disable_cli_model_fetch_with_model_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        disable_cli_model_fetch=True,
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    with pytest.raises(AppIncompatibleError, match="downloads its model at runtime"):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_disable_cli_model_fetch_without_model_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        disable_cli_model_fetch=True,
    )
    app = App(info)
    result = app.fetch(tmp_path)

    assert result == tmp_path / "test_app"
    assert result.exists()


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
        MagicMock(side_effect=AssetNotFoundError("not found")),
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


def test_fetch_downloaded_metadata_missing_keys_raises(monkeypatch, tmp_path):
    """A downloaded asset missing required metadata keys points at filing an issue."""
    # metadata.json without the required model_id field.
    bad_meta = _make_fake_download(model_files=["model.onnx"], model_id=None)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", bad_meta)
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
    with pytest.raises(AppIncompatibleError, match="please file an issue"):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_downloaded_model_id_mismatch_raises(monkeypatch, tmp_path):
    """A downloaded asset whose metadata model_id differs from the request is a bug."""
    wrong_id = _make_fake_download(model_files=["model.onnx"], model_id="other_model")
    monkeypatch.setattr("qai_hub_apps.registry.base.download", wrong_id)
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
    with pytest.raises(AppIncompatibleError, match="reports model id 'other_model'"):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_unknown_model_asset_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(side_effect=KeyError("No model exists")),
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

    def fail_model_download(
        url: str, path: Path, extract: bool = False, quiet: bool = False
    ) -> Path:
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
        MagicMock(side_effect=AssetNotFoundError("not found")),
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
    one_file = _make_fake_download(model_files=["model.onnx"], extra_files=[])
    monkeypatch.setattr("qai_hub_apps.registry.base.download", one_file)
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
    no_metadata = _make_fake_download(model_files=["model.onnx"], with_metadata=False)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", no_metadata)
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
    with pytest.raises(AppIncompatibleError, match=r"is missing metadata\.json"):
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


def test_fetch_dest_exists_overwrite_replaces_in_place(monkeypatch, tmp_path):
    existing = tmp_path / "test_app"
    existing.mkdir()
    (existing / "stale.txt").write_text("old")
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)

    info = make_app_info(url=AppUrl(source="https://example.com/app.zip"))
    app = App(info)
    result = app.fetch(tmp_path, overwrite=True)

    assert result == tmp_path / "test_app"
    assert result.exists()
    assert not (result / "stale.txt").exists()


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


def test_detail_fields_includes_supported_devices():
    app = App(make_app_info(supported_devices=["Device A", "Device B"]))
    fields = dict(app.detail_fields())
    assert fields["Supported Devices"] == "Device A, Device B"


def test_registry_apps_returns_all(sample_registry_yaml):
    registry = Registry.load(sample_registry_yaml)
    apps = list(registry.apps)
    assert len(apps) == 1
    assert apps[0].id == "test_app"


def test_deprecation_message_none_when_not_deprecated():
    app = App(make_app_info(status="published"))
    assert app.deprecation_message() is None


def test_deprecation_message_default_when_no_notice():
    app = App(make_app_info(status="deprecated"))
    assert app.deprecation_message() == DEFAULT_DEPRECATION_MESSAGE


def test_deprecation_message_uses_notice_when_set():
    app = App(make_app_info(status="deprecated", deprecation_notice="Use foo instead."))
    assert app.deprecation_message() == "Use foo instead."


def test_repr_shows_deprecation_banner():
    app = App(make_app_info(status="deprecated"))
    assert f"DEPRECATED: {DEFAULT_DEPRECATION_MESSAGE}" in repr(app)


def test_repr_omits_deprecation_banner_when_not_deprecated():
    app = App(make_app_info(status="published"))
    assert "DEPRECATED" not in repr(app)


def test_fetch_app_deprecated_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: True)
    monkeypatch.setattr("qai_hub_apps.registry.base.is_app_supported", lambda app: True)

    from qai_hub_apps.configs.app_yaml import AppUrl
    from qai_hub_apps.configs.registry_yaml import AppRegistry

    info = make_app_info(
        status="deprecated",
        deprecation_notice="Use foo instead.",
        url=AppUrl(source="https://example.com/app.zip"),
    )
    raw = AppRegistry(schema_version="1.1", min_cli_version="0.0.1", apps=[info])
    registry = Registry(raw)
    registry.fetch_app("test_app", tmp_path)

    assert "Use foo instead." in caplog.text


def test_fetch_ambiguous_model_id_and_path_raises(monkeypatch, tmp_path):
    """A --model value that is both a supported model and an existing path is rejected."""
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_dir="models",
    )
    app = App(info)
    # Auto-resolved --model enters as a relative path equal to a supported model id.
    with pytest.raises(InvalidArgumentError, match="--model-id or --model-path"):
        app.fetch(tmp_path / "out", model_asset=ModelAsset(path=Path("test_model")))


def test_fetch_local_export_dir_places_files(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    # get_asset_url must NOT be called for a local export.
    no_call = MagicMock(
        side_effect=AssertionError("get_asset_url should not be called")
    )
    monkeypatch.setattr("qai_hub_apps.registry.base.get_asset_url", no_call)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    export = _make_export_dir(tmp_path, model_files=["model1.onnx", "model2.onnx"])
    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/a.onnx", "models/b.onnx"],
    )
    app = App(info)
    result = app.fetch(tmp_path / "out", model_asset=ModelAsset(path=export))

    assert (result / "models" / "a.onnx").exists()
    assert (result / "models" / "b.onnx").exists()
    # Source export is left untouched (copied).
    assert (export / "model1.onnx").exists()


def test_fetch_local_export_zip_places_files(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    zip_path = _make_export_dir(tmp_path, model_files=["model1.onnx"], as_zip=True)
    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_dir="models",
    )
    app = App(info)
    result = app.fetch(tmp_path / "out", model_asset=ModelAsset(path=zip_path))

    assert (result / "models" / "model1.onnx").exists()
    assert (result / "models" / "metadata.json").exists()


def test_fetch_local_export_unsupported_model_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    export = _make_export_dir(
        tmp_path, model_files=["model1.onnx"], model_id="other_model"
    )
    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_dir="models",
    )
    app = App(info)
    with pytest.raises(AppIncompatibleError, match="other_model"):
        app.fetch(tmp_path / "out", model_asset=ModelAsset(path=export))


def test_fetch_local_export_no_model_id_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    export = _make_export_dir(tmp_path, model_files=["model1.onnx"], model_id=None)
    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_dir="models",
    )
    app = App(info)
    with pytest.raises(AppIncompatibleError, match="model_id"):
        app.fetch(tmp_path / "out", model_asset=ModelAsset(path=export))


def test_fetch_local_export_missing_metadata_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    export = _make_export_dir(
        tmp_path, model_files=["model1.onnx"], with_metadata=False
    )
    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_dir="models",
    )
    app = App(info)
    with pytest.raises(AppIncompatibleError, match=r"metadata.json"):
        app.fetch(tmp_path / "out", model_asset=ModelAsset(path=export))


def test_fetch_local_export_nonexistent_path_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_dir="models",
    )
    app = App(info)
    missing = tmp_path / "does_not_exist"
    with pytest.raises(AppIncompatibleError, match=r"directory or a .zip"):
        app.fetch(tmp_path / "out", model_asset=ModelAsset(path=missing))


def test_fetch_local_export_plain_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    plain = tmp_path / "model.onnx"
    plain.touch()
    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_dir="models",
    )
    app = App(info)
    with pytest.raises(AppIncompatibleError, match=r"directory or a .zip"):
        app.fetch(tmp_path / "out", model_asset=ModelAsset(path=plain))


def test_fetch_model_sharing_restricted_hint(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    err = AssetNotFoundError("No assets for restricted_model")
    err.model_sharing_restricted = True
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(side_effect=err),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_dir="models",
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset=None)
    with pytest.raises(ModelAssetNotFoundError, match="--model <exported_model_path>"):
        app.fetch(tmp_path / "out", model_asset=asset)


def test_fetch_app_unsupported_platform_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: True)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.is_app_supported", lambda app: False
    )

    from qai_hub_apps.configs.app_yaml import AppUrl
    from qai_hub_apps.configs.registry_yaml import AppRegistry

    info = make_app_info(url=AppUrl(source="https://example.com/app.zip"))
    raw = AppRegistry(schema_version="1.1", min_cli_version="0.0.1", apps=[info])
    registry = Registry(raw)
    registry.fetch_app("test_app", tmp_path)

    assert "This app may not be supported on the current device." in caplog.text


def test_supported_chipsets_unknown_device_raises_with_issue_url(monkeypatch):
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.device_to_chipset",
        MagicMock(side_effect=KeyError("Bogus")),
    )
    app = App(make_app_info(supported_devices=["Bogus"]))
    with pytest.raises(AppIncompatibleError, match="known AI Hub device"):
        _ = app.supported_chipsets


def test_ensure_chipset_supported_noop_without_devices(monkeypatch):
    # An app with no supported_devices places no restriction on --chipset.
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.device_to_chipset",
        MagicMock(side_effect=AssertionError("should not resolve")),
    )
    app = App(make_app_info(supported_devices=[]))
    app._ensure_chipset_supported("any-chipset")  # no raise


def test_ensure_device_supported_noop_without_devices():
    # An app with no supported_devices places no restriction on --device.
    app = App(make_app_info(supported_devices=[]))
    app._ensure_device_supported("any-device")  # no raise


def test_fetch_unsupported_chipset_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.device_to_chipset", lambda d: "chip-1"
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/model.onnx"],
        supported_devices=["Device A"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset="chip-2")
    with pytest.raises(AppIncompatibleError, match="Chipset 'chip-2' is not supported"):
        app.fetch(tmp_path, model_asset=asset)


def test_fetch_supported_chipset_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.device_to_chipset", lambda d: "chip-1"
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_dir="models",
        supported_devices=["Device A"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", chipset="chip-1")
    result = app.fetch(tmp_path, model_asset=asset)
    assert result == tmp_path / "test_app"


def test_fetch_unsupported_device_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        MagicMock(return_value="https://example.com/model.zip"),
    )

    info = make_app_info(
        url=AppUrl(source="https://example.com/app.zip"),
        related_models=["test_model"],
        model_file_paths=["models/model.onnx"],
        supported_devices=["Device A"],
    )
    app = App(info)
    asset = ModelAsset(model_id="test_model", device="Device B")
    with pytest.raises(
        AppIncompatibleError, match="Device 'Device B' is not supported"
    ):
        app.fetch(tmp_path, model_asset=asset)
