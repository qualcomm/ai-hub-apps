# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import qai_hub_apps_test.scripts.generate_registry as gen_mod
from qai_hub_apps_test.configs.info_yaml import AppLanguage
from qai_hub_apps_test.configs.registry_yaml import AppRegistry
from qai_hub_apps_test.conftest import make_sample_app_info
from qai_hub_apps_test.scripts.generate_registry import (
    _resolve_repo_url,
    generate_registry,
    upload_app,
    upload_registry,
)

pytestmark = pytest.mark.bundler_unit

_REPO_BASE = "https://github.com/qualcomm/ai-hub-apps"
_CLI_VERSION = "0.27.0"


def _fake_attempt(fn: Callable[[], None]) -> None:
    fn()


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
        == f"{_REPO_BASE}/tree/v1.0/apps/my_app"
    )


def test_upload_registry_s3_key_format(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("schema_version: '1.0'\n")

    bucket = MagicMock()
    captured_key: list[str] = []

    with patch.object(
        gen_mod, "attempt_with_s3_credentials_warning", side_effect=_fake_attempt
    ):
        bucket.upload_file.side_effect = lambda path, key, **kw: captured_key.append(
            key
        )
        upload_registry(registry_path, bucket, "qai-hub-apps/releases", _CLI_VERSION)

    assert captured_key == [f"qai-hub-apps/releases/{_CLI_VERSION}/registry.yaml"]


def test_upload_app_s3_key_format(tmp_path: Path) -> None:
    zip_path = tmp_path / "myapp.zip"
    zip_path.write_bytes(b"PK")

    bucket = MagicMock()
    captured_key: list[str] = []

    with patch.object(
        gen_mod, "attempt_with_s3_credentials_warning", side_effect=_fake_attempt
    ):
        bucket.upload_file.side_effect = lambda path, key, **kw: captured_key.append(
            key
        )
        upload_app(zip_path, "myapp", bucket, "qai-hub-apps/releases", _CLI_VERSION)

    assert captured_key == [f"qai-hub-apps/releases/{_CLI_VERSION}/myapp/source.zip"]


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


def test_include_all_includes_unpublished_and_excluded(tmp_path: Path) -> None:
    pub_dir = tmp_path / "pub_app"
    pub_dir.mkdir()
    unpub_dir = tmp_path / "unpub_app"
    unpub_dir.mkdir()
    excluded_dir = tmp_path / "excluded_app"
    excluded_dir.mkdir()
    apps = [
        (make_sample_app_info(id="pub_app", status="published"), pub_dir),
        (make_sample_app_info(id="unpub_app", status="unpublished"), unpub_dir),
        (
            make_sample_app_info(
                id="excluded_app", status="published", include_in_cli=False
            ),
            excluded_dir,
        ),
    ]
    generate_registry(
        tmp_path, apps, _REPO_BASE, "main", _CLI_VERSION, include_all=True
    )

    registry = AppRegistry.from_yaml(tmp_path / "registry.yaml")
    assert {a.id for a in registry.apps} == {"pub_app", "unpub_app", "excluded_app"}


def test_default_excludes_unpublished_and_excluded(tmp_path: Path) -> None:
    pub_dir = tmp_path / "pub_app"
    pub_dir.mkdir()
    unpub_dir = tmp_path / "unpub_app"
    unpub_dir.mkdir()
    excluded_dir = tmp_path / "excluded_app"
    excluded_dir.mkdir()
    apps = [
        (make_sample_app_info(id="pub_app", status="published"), pub_dir),
        (make_sample_app_info(id="unpub_app", status="unpublished"), unpub_dir),
        (
            make_sample_app_info(
                id="excluded_app", status="published", include_in_cli=False
            ),
            excluded_dir,
        ),
    ]
    generate_registry(tmp_path, apps, _REPO_BASE, "main", _CLI_VERSION)

    registry = AppRegistry.from_yaml(tmp_path / "registry.yaml")
    assert {a.id for a in registry.apps} == {"pub_app"}


def test_include_all_with_build_and_upload_raises(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    apps = [(make_sample_app_info(id="myapp", status="published"), app_dir)]
    with pytest.raises(SystemExit, match="include_all cannot be used"):
        generate_registry(
            tmp_path,
            apps,
            _REPO_BASE,
            "main",
            _CLI_VERSION,
            build_and_upload=True,
            include_all=True,
        )


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
