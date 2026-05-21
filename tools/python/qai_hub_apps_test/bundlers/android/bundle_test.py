# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import qai_hub_apps_test.bundlers.android.bundle as android_bundle_mod
from qai_hub_apps_test.bundlers.android.bundle import bundle_source
from qai_hub_apps_test.conftest import FAKE_VERSIONS

pytestmark = pytest.mark.bundler_unit


def test_regular_file_copied(dummy_android_app_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    with patch.object(android_bundle_mod, "load_versions", return_value=FAKE_VERSIONS):
        bundle_source(dummy_android_app_path, out)
    assert (out / "build.gradle").exists()


def test_symlinked_directory_resolved(
    dummy_android_app_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "bundle"
    with patch.object(android_bundle_mod, "load_versions", return_value=FAKE_VERSIONS):
        bundle_source(dummy_android_app_path, out)

    tflite_out = out / "src" / "main" / "java" / "com" / "quicinc" / "tflite"
    assert tflite_out.exists()
    assert not tflite_out.is_symlink()
    assert (tflite_out / "TFLiteHelpers.java").read_text() == "// TFLite helpers\n"


def test_symlinked_file_resolved(dummy_android_app_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    with patch.object(android_bundle_mod, "load_versions", return_value=FAKE_VERSIONS):
        bundle_source(dummy_android_app_path, out)

    image_proc = (
        out / "src" / "main" / "java" / "com" / "quicinc" / "ImageProcessing.java"
    )
    assert image_proc.exists()
    assert not image_proc.is_symlink()
    assert image_proc.read_text() == "// image processing\n"


def test_no_symlinks_in_output(dummy_android_app_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    with patch.object(android_bundle_mod, "load_versions", return_value=FAKE_VERSIONS):
        bundle_source(dummy_android_app_path, out)

    symlinks = [p for p in out.rglob("*") if p.is_symlink()]
    assert symlinks == [], f"Unexpected symlinks in bundle: {symlinks}"


def test_versions_inlined(dummy_android_app_path: Path, tmp_path: Path) -> None:
    """${VAR} and bare VAR references are replaced with resolved values."""
    out = tmp_path / "bundle"
    with patch.object(android_bundle_mod, "load_versions", return_value=FAKE_VERSIONS):
        bundle_source(dummy_android_app_path, out)

    content = (out / "build.gradle").read_text()
    assert "${TF_LITE_VERSION}" not in content
    assert FAKE_VERSIONS["TF_LITE_VERSION"] in content
    assert "${QNN_VERSION}" not in content
    assert FAKE_VERSIONS["QNN_VERSION"] in content
    assert "ANDROID_COMPILE_API" not in content
    assert "ANDROID_NDK_VERSION" not in content
    assert FAKE_VERSIONS["ANDROID_NDK_VERSION"] in content


def test_common_gradle_emptied(dummy_android_app_path: Path, tmp_path: Path) -> None:
    """common.gradle in bundle is emptied so apply from succeeds but does nothing."""
    out = tmp_path / "bundle"
    with patch.object(android_bundle_mod, "load_versions", return_value=FAKE_VERSIONS):
        bundle_source(dummy_android_app_path, out)

    assert (out / "_shared" / "android" / "common.gradle").read_text() == ""
