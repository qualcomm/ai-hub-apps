# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pytest

from qai_hub_apps_test.bundlers.python.requirements import (
    _pkg_name,
    merge_requirements,
    read_module_requirements,
)

pytestmark = pytest.mark.bundler_unit


def test_plain_name() -> None:
    assert _pkg_name("numpy") == "numpy"


@pytest.mark.parametrize(
    "req",
    ["numpy==1.24", "numpy>=2.0", "numpy<=2.0", "numpy!=1.0", "numpy~=2.0"],
)
def test_version_specifiers(req: str) -> None:
    assert _pkg_name(req) == "numpy"


def test_extras_stripped() -> None:
    assert _pkg_name("boto3[s3]") == "boto3"


def test_normalizes_underscore_separator() -> None:
    assert _pkg_name("my_pkg") == "my-pkg"


def test_normalizes_dot_separator() -> None:
    assert _pkg_name("my.pkg") == "my-pkg"


def test_lowercases() -> None:
    assert _pkg_name("MyPkg") == "mypkg"


def test_complex() -> None:
    assert _pkg_name("My.SDK_utils>=2.0") == "my-sdk-utils"


def test_reads_existing_requirements_file(tmp_path: Path) -> None:
    req_dir = tmp_path / "requirements"
    req_dir.mkdir()
    (req_dir / "requirements-draw.txt").write_text("pillow>=9.0\nnumpy\n")
    draw_py = tmp_path / "draw.py"
    draw_py.write_text("")

    result = read_module_requirements(draw_py)
    assert result == ["pillow>=9.0", "numpy"]


def test_strips_comments_and_blank_lines(tmp_path: Path) -> None:
    req_dir = tmp_path / "requirements"
    req_dir.mkdir()
    (req_dir / "requirements-draw.txt").write_text(
        "# this is a comment\n\npillow>=9.0\n\n"
    )
    draw_py = tmp_path / "draw.py"
    draw_py.write_text("")

    result = read_module_requirements(draw_py)
    assert result == ["pillow>=9.0"]


def test_missing_requirements_file_warns_and_returns_empty(tmp_path: Path) -> None:
    draw_py = tmp_path / "draw.py"
    draw_py.write_text("")

    with pytest.warns(UserWarning, match="No requirements file found"):
        result = read_module_requirements(draw_py)
    assert result == []


def test_app_reqs_only(tmp_path: Path) -> None:
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("numpy\npillow\n")
    result = merge_requirements(req_file, [])
    assert "numpy" in result
    assert "pillow" in result


def test_utils_reqs_only(tmp_path: Path) -> None:
    req_file = tmp_path / "requirements.txt"
    result = merge_requirements(req_file, ["torch>=2.0"])
    assert "torch>=2.0" in result


def test_no_duplicates(tmp_path: Path) -> None:
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("numpy==2.0\n")
    result = merge_requirements(req_file, ["numpy==2.0"])
    assert result.count("numpy==2.0") == 1


def test_conflicting_versions_warns_and_includes_both(tmp_path: Path) -> None:
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("numpy==1.24\n")
    with pytest.warns(UserWarning, match="multiple specifiers"):
        result = merge_requirements(req_file, ["numpy>=2.0"])
    assert "numpy==1.24" in result
    assert "numpy>=2.0" in result


def test_empty_file_and_no_utils(tmp_path: Path) -> None:
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("")
    assert merge_requirements(req_file, []) == []


def test_comments_and_blank_lines_ignored(tmp_path: Path) -> None:
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("# comment\n\nnumpy\n")
    result = merge_requirements(req_file, [])
    assert result == ["numpy"]


def test_missing_requirements_file_returns_utils_reqs(tmp_path: Path) -> None:
    req_file = tmp_path / "requirements.txt"
    result = merge_requirements(req_file, ["torch"])
    assert result == ["torch"]
