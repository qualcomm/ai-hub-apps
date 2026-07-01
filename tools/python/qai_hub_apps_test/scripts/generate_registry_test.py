# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import qai_hub_apps_test.scripts.generate_registry as gen_mod
from qai_hub_apps_test.configs.info_yaml import AppLanguage, AppStatus, QAIHAAppInfo
from qai_hub_apps_test.configs.registry_yaml import AppRegistry
from qai_hub_apps_test.conftest import make_sample_app_info
from qai_hub_apps_test.scripts.generate_registry import (
    RegistryScope,
    _resolve_repo_url,
    generate_registry,
    upload_app,
    upload_registry,
)

pytestmark = pytest.mark.bundler_unit

_REPO_BASE = "https://github.com/qualcomm/ai-hub-apps"
_CLI_VERSION = "0.27.0"


def _make_one_app_per_status(
    tmp_path: Path,
) -> list[tuple[QAIHAAppInfo, Path]]:
    """One app per AppStatus, each in a directory named after its status value."""
    apps = []
    for status in AppStatus:
        app_dir = tmp_path / status.value
        app_dir.mkdir()
        apps.append(
            (make_sample_app_info(id=status.value, status=status.value), app_dir)
        )
    return apps


def test_uses_app_repo_url_when_set() -> None:
    info = make_sample_app_info(app_repo_url="https://github.com/external/repo")
    assert (
        _resolve_repo_url(info, _REPO_BASE, "main")
        == "https://github.com/external/repo"
    )


def test_constructs_url_from_relative_path() -> None:
    info = make_sample_app_info(app_repo_url=None, app_repo_relative_path="my_app")
    assert (
        _resolve_repo_url(info, _REPO_BASE, "v1.0")
        == f"{_REPO_BASE}/tree/apps/v1.0/my_app"
    )


def test_upload_registry_s3_key_format(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("schema_version: '1.0'\n")

    bucket = MagicMock()
    upload_registry(registry_path, bucket, "qai-hub-apps/releases", _CLI_VERSION)

    assert bucket.upload_file.call_count == 1
    assert (
        bucket.upload_file.call_args.args[1]
        == f"qai-hub-apps/releases/{_CLI_VERSION}/registry.yaml"
    )


def test_upload_app_s3_key_format(tmp_path: Path) -> None:
    zip_path = tmp_path / "myapp.zip"
    zip_path.write_bytes(b"PK")

    bucket = MagicMock()
    upload_app(zip_path, "myapp", bucket, "qai-hub-apps/releases", _CLI_VERSION)

    assert bucket.upload_file.call_count == 1
    assert (
        bucket.upload_file.call_args.args[1]
        == f"qai-hub-apps/releases/{_CLI_VERSION}/myapp/source.zip"
    )


def test_no_build_writes_registry_yaml(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    apps = [(make_sample_app_info(id="myapp", status="published"), app_dir)]
    generate_registry(tmp_path, apps, _REPO_BASE, "main", _CLI_VERSION)
    assert (tmp_path / "registry.yaml").exists()


def test_skips_unpublished_apps(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    apps = [(make_sample_app_info(id="myapp", status="unpublished"), app_dir)]
    generate_registry(tmp_path, apps, _REPO_BASE, "main", _CLI_VERSION)

    registry = AppRegistry.from_yaml(tmp_path / "registry.yaml")
    assert len(registry.apps) == 0


def test_skips_non_python_apps(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    apps = [
        (
            make_sample_app_info(
                id="myapp", status="published", languages=[AppLanguage.CPP]
            ),
            app_dir,
        )
    ]
    generate_registry(tmp_path, apps, _REPO_BASE, "main", _CLI_VERSION)

    registry = AppRegistry.from_yaml(tmp_path / "registry.yaml")
    assert len(registry.apps) == 0


def test_production_set_includes_published_and_deprecated(tmp_path: Path) -> None:
    generate_registry(
        tmp_path,
        _make_one_app_per_status(tmp_path),
        _REPO_BASE,
        "main",
        _CLI_VERSION,
        scope=RegistryScope.PRODUCTION,
    )
    registry = AppRegistry.from_yaml(tmp_path / "registry.yaml")
    assert {a.status for a in registry.apps} == {
        AppStatus.PUBLISHED,
        AppStatus.DEPRECATED,
    }


def test_default_set_is_production(tmp_path: Path) -> None:
    generate_registry(
        tmp_path, _make_one_app_per_status(tmp_path), _REPO_BASE, "main", _CLI_VERSION
    )
    registry = AppRegistry.from_yaml(tmp_path / "registry.yaml")
    assert {a.status for a in registry.apps} == {
        AppStatus.PUBLISHED,
        AppStatus.DEPRECATED,
    }


def test_test_set_excludes_website_only(tmp_path: Path) -> None:
    generate_registry(
        tmp_path,
        _make_one_app_per_status(tmp_path),
        _REPO_BASE,
        "main",
        _CLI_VERSION,
        scope=RegistryScope.TEST,
    )
    registry = AppRegistry.from_yaml(tmp_path / "registry.yaml")
    assert {a.status for a in registry.apps} == {
        AppStatus.UNPUBLISHED,
        AppStatus.PUBLISHED,
        AppStatus.DEPRECATED,
    }


def test_all_set_includes_every_app(tmp_path: Path) -> None:
    generate_registry(
        tmp_path,
        _make_one_app_per_status(tmp_path),
        _REPO_BASE,
        "main",
        _CLI_VERSION,
        scope=RegistryScope.ALL,
    )
    registry = AppRegistry.from_yaml(tmp_path / "registry.yaml")
    assert {a.status for a in registry.apps} == set(AppStatus)


@pytest.mark.parametrize("scope", [RegistryScope.TEST, RegistryScope.ALL])
def test_non_production_scope_with_build_and_upload_raises(
    tmp_path: Path, scope: RegistryScope
) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    apps = [(make_sample_app_info(id="myapp", status="published"), app_dir)]
    with pytest.raises(SystemExit, match="build_and_upload requires"):
        generate_registry(
            tmp_path,
            apps,
            _REPO_BASE,
            "main",
            _CLI_VERSION,
            build_and_upload=True,
            scope=scope,
        )


def test_dev_version_upload_fails_without_force(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    apps = [(make_sample_app_info(id="myapp", status="published"), app_dir)]
    with pytest.raises(SystemExit, match="development build"):
        generate_registry(
            tmp_path,
            apps,
            _REPO_BASE,
            "main",
            "0.32.0.dev27+gabc1234",
            build_and_upload=True,
        )


def test_dev_version_force_uploads_to_dev_prefix(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()

    apps = [(make_sample_app_info(id="myapp", status="unpublished"), app_dir)]
    with (
        patch.object(gen_mod, "get_qaihm_s3", return_value=(MagicMock(), None)),
        patch.object(gen_mod, "bundle_app"),
        patch.object(gen_mod, "upload_app") as mock_upload_app,
        patch.object(gen_mod, "upload_registry") as mock_upload_registry,
    ):
        generate_registry(
            tmp_path,
            apps,
            _REPO_BASE,
            "main",
            "0.32.0.dev27+gabc1234",
            build_and_upload=True,
            scope=RegistryScope.TEST,
            force=True,
        )

    registry = AppRegistry.from_yaml(tmp_path / "registry.yaml")
    assert {a.id for a in registry.apps} == {"myapp"}

    # Dev versions upload under the dedicated "dev" subfolder.
    dev_prefix = "qai-hub-apps/releases/dev"
    assert mock_upload_app.call_args.args[3] == dev_prefix
    assert mock_upload_registry.call_args.args[2] == dev_prefix
    app_url = next(iter(registry.apps)).url
    assert app_url is not None
    assert f"/{dev_prefix}/" in app_url.source


def test_raises_on_duplicate_app_ids(tmp_path: Path) -> None:
    app_dir1 = tmp_path / "myapp"
    app_dir1.mkdir()
    app_dir2 = tmp_path / "myapp2"
    app_dir2.mkdir()
    apps = [
        (make_sample_app_info(id="myapp", status="published"), app_dir1),
        (make_sample_app_info(id="myapp", status="published"), app_dir2),
    ]
    with pytest.raises(SystemExit):
        generate_registry(tmp_path, apps, _REPO_BASE, "main", _CLI_VERSION)


def test_raises_on_id_directory_mismatch(tmp_path: Path) -> None:
    app_dir = tmp_path / "wrong_dir_name"
    app_dir.mkdir()
    apps = [(make_sample_app_info(id="myapp", status="published"), app_dir)]
    with pytest.raises(SystemExit):
        generate_registry(tmp_path, apps, _REPO_BASE, "main", _CLI_VERSION)
