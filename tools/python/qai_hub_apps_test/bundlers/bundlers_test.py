# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import qai_hub_apps_test.bundlers as bundlers_mod
from qai_hub_apps_test.bundlers import bundle_app
from qai_hub_apps_test.configs.info_yaml import AppLanguage
from qai_hub_apps_test.conftest import FAKE_VERSIONS, make_sample_app_info

pytestmark = pytest.mark.bundler_unit

_UTILS = "qai_hub_apps_utils"


def test_bundle_app_non_python_raises(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    make_sample_app_info(id="myapp", languages=[AppLanguage.CPP]).to_yaml(
        app_dir / "info.yaml", write_if_empty=True
    )
    with pytest.raises(NotImplementedError):
        bundle_app(app_dir, tmp_path / "out")


def test_bundle_app_by_str_id_resolves_dir(
    dummy_python_app_path: Path,
    dummy_python_utils_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mock_bundle_source(
        app_root: Path,
        out_dir: Path,
        utils_parent: Path,
        shared_scripts_root: Path | None = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bundlers_mod, "_bundle_python_source", mock_bundle_source)
    monkeypatch.setattr(bundlers_mod, "find_app_dir", lambda _: dummy_python_app_path)

    out = tmp_path / "out"
    bundle_app("my_dummy_app", out, utils_parent=dummy_python_utils_path)

    assert (out / "my_dummy_app").is_dir()


def test_bundle_python_app_e2e(
    dummy_python_app_path: Path,
    dummy_python_utils_path: Path,
    dummy_scripts_path: Path,
    tmp_path: Path,
) -> None:
    """E2E: bundle a Python app with utils imports and install scripts (no mocks).

    Verifies that bundle_app:
    - copies app source files
    - resolves and copies the directly imported qai_hub_apps_utils module
    - follows transitive utils imports (helper -> math_utils)
    - includes the qai_hub_apps_utils package __init__.py
    - does NOT copy unreferenced utils modules
    - merges app requirements with per-module utils requirements
    - copies referenced shared scripts to scripts/
    - copies versions.env to scripts/
    - rewrites source lines in install_*.sh
    """
    # Add an install script to the app
    (dummy_python_app_path / "install_runtime.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"source {dummy_scripts_path}/apt_utils.sh\n"
        "install_apt_pkg libfoo\n"
    )

    out_dir = tmp_path / "out"
    bundle_app(
        dummy_python_app_path,
        out_dir,
        utils_parent=dummy_python_utils_path,
        shared_scripts_root=dummy_scripts_path,
    )

    bundle = out_dir / "my_dummy_app"
    assert bundle.is_dir()

    # app source
    assert (bundle / "main.py").exists()

    # directly imported qai_hub_apps_utils module + package init
    assert (bundle / _UTILS / "__init__.py").exists()
    assert (bundle / _UTILS / "helper.py").exists()

    # transitively imported qai_hub_apps_utils module
    assert (bundle / _UTILS / "math_utils.py").exists()

    # unreferenced module NOT copied
    assert not (bundle / _UTILS / "unreferenced.py").exists()

    # merged requirements: app dep + per-module utils dep
    reqs = (bundle / "requirements.txt").read_text()
    assert "Pillow>=9.0" in reqs
    assert "numpy>=1.24" in reqs

    # shared scripts copied transitively (apt_utils + load_versions)
    assert (bundle / "scripts" / "apt_utils.sh").exists()
    assert (bundle / "scripts" / "load_versions.sh").exists()
    assert (bundle / "scripts" / "versions.env").exists()

    # source line rewritten to bundle-local path
    assert (
        'source "$(dirname "${BASH_SOURCE[0]}")/scripts/apt_utils.sh"'
        in (bundle / "install_runtime.sh").read_text()
    )


def test_bundle_android_app_e2e(
    dummy_android_app_path: Path,
    dummy_scripts_path: Path,
    tmp_path: Path,
) -> None:
    """E2E: bundle an Android app with symlinked shared code and version variables.

    Verifies that bundle_app:
    - resolves symlinks (tflite/ dir, ImageProcessing.java)
    - copies shared scripts and versions.env to scripts/
    - rewrites source lines in install_build.sh
    - inlines version variables in build.gradle
    - empties common.gradle
    """
    out_dir = tmp_path / "out"
    bundle_app(
        dummy_android_app_path,
        out_dir,
        shared_scripts_root=dummy_scripts_path,
    )

    bundle = out_dir / "my_dummy_android_app"
    assert bundle.is_dir()

    # symlinks resolved
    tflite = bundle / "src" / "main" / "java" / "com" / "quicinc" / "tflite"
    assert tflite.exists() and not tflite.is_symlink()
    assert (tflite / "TFLiteHelpers.java").exists()
    image_proc = (
        bundle / "src" / "main" / "java" / "com" / "quicinc" / "ImageProcessing.java"
    )
    assert image_proc.exists() and not image_proc.is_symlink()

    # versions.env copied by shell bundler
    assert (bundle / "scripts" / "versions.env").exists()

    # source line in install_build.sh rewritten to bundle-local path
    install_build = (bundle / "install_build.sh").read_text()
    assert (
        'source "$(dirname "${BASH_SOURCE[0]}")/scripts/android_utils.sh"'
        in install_build
    )

    # version variables inlined in build.gradle
    gradle = (bundle / "build.gradle").read_text()
    assert "${TF_LITE_VERSION}" not in gradle
    assert FAKE_VERSIONS["TF_LITE_VERSION"] in gradle
    assert "ANDROID_COMPILE_API" not in gradle

    # common.gradle emptied
    assert (bundle / "_shared" / "android" / "common.gradle").read_text() == ""


def test_bundle_app_make_zip(
    dummy_python_app_path: Path, dummy_python_utils_path: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    bundle_app(
        dummy_python_app_path,
        out_dir,
        utils_parent=dummy_python_utils_path,
        make_zip=True,
    )
    zip_path = out_dir / "my_dummy_app.zip"
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        assert "main.py" in zf.namelist()


def test_bundle_app_overwrites_existing_dest(
    dummy_python_app_path: Path, dummy_python_utils_path: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    bundle_app(dummy_python_app_path, out_dir, utils_parent=dummy_python_utils_path)
    (dummy_python_app_path / "main.py").write_text("# v2\n")
    bundle_app(dummy_python_app_path, out_dir, utils_parent=dummy_python_utils_path)
    assert "v2" in (out_dir / "my_dummy_app" / "main.py").read_text()


def test_bundle_app_includes_dockerfile(
    dummy_python_app_path: Path,
    dummy_python_utils_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_sample_app_info(id="my_dummy_app", base_docker="ubuntu.dockerfile").to_yaml(
        dummy_python_app_path / "info.yaml", write_if_empty=True
    )

    fake_docker_root = tmp_path / "fakerepo" / "tools" / "docker"
    fake_docker_root.mkdir(parents=True)
    (fake_docker_root / "ubuntu.dockerfile").write_text("FROM ubuntu:24.04\n")
    monkeypatch.setattr(bundlers_mod, "DOCKER_ROOT", fake_docker_root)

    out_dir = tmp_path / "out"
    bundle_app(dummy_python_app_path, out_dir, utils_parent=dummy_python_utils_path)

    dockerfile = out_dir / "my_dummy_app" / "Dockerfile"
    assert dockerfile.is_file()
    assert "FROM ubuntu:24.04" in dockerfile.read_text()


def test_bundle_app_no_dockerfile_when_base_docker_unset(
    dummy_python_app_path: Path,
    dummy_python_utils_path: Path,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    bundle_app(dummy_python_app_path, out_dir, utils_parent=dummy_python_utils_path)
    assert not (out_dir / "my_dummy_app" / "Dockerfile").exists()


def test_bundle_app_missing_dockerfile_raises(
    dummy_python_app_path: Path,
    dummy_python_utils_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_sample_app_info(
        id="my_dummy_app", base_docker="nonexistent.dockerfile"
    ).to_yaml(dummy_python_app_path / "info.yaml", write_if_empty=True)

    fake_docker_root = tmp_path / "fakerepo" / "tools" / "docker"
    fake_docker_root.mkdir(parents=True)
    monkeypatch.setattr(bundlers_mod, "DOCKER_ROOT", fake_docker_root)

    with pytest.raises(FileNotFoundError, match=r"nonexistent.dockerfile"):
        bundle_app(
            dummy_python_app_path,
            tmp_path / "out",
            utils_parent=dummy_python_utils_path,
        )
