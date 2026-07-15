# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pytest

from qai_hub_apps_test.bundlers.python.utils_collector import (
    collect_all_utils_files,
    collect_utils_imports_from_file,
    init_files_for_utils_file,
    module_to_utils_file,
)

pytestmark = pytest.mark.bundler_unit

_UTILS = "qai_hub_apps_utils"


def test_no_utils_imports(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text("import os\nimport sys\n")
    assert collect_utils_imports_from_file(f) == set()


def test_direct_utils_import(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text(f"import {_UTILS}\n")
    assert collect_utils_imports_from_file(f) == {_UTILS}


def test_from_submodule_import(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text(f"from {_UTILS}.draw import something\n")
    assert collect_utils_imports_from_file(f) == {f"{_UTILS}.draw"}


def test_submodule_direct_import(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text(f"import {_UTILS}.image_processing\n")
    assert collect_utils_imports_from_file(f) == {f"{_UTILS}.image_processing"}


def test_syntax_error_warns_and_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "broken.py"
    f.write_text("def foo(\n")
    with pytest.warns(UserWarning, match="Could not parse"):
        result = collect_utils_imports_from_file(f)
    assert result == set()


def test_multiple_utils_imports(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text(
        f"from {_UTILS}.draw import draw_boxes\n"
        f"import {_UTILS}.image_processing\n"
        "import os\n"
    )
    assert collect_utils_imports_from_file(f) == {
        f"{_UTILS}.draw",
        f"{_UTILS}.image_processing",
    }


def test_resolves_module_file(tmp_path: Path) -> None:
    utils_dir = tmp_path / _UTILS
    utils_dir.mkdir()
    draw = utils_dir / "draw.py"
    draw.write_text("")
    result = module_to_utils_file(f"{_UTILS}.draw", tmp_path)
    assert result == draw


def test_resolves_package_init(tmp_path: Path) -> None:
    utils_dir = tmp_path / _UTILS
    utils_dir.mkdir()
    init = utils_dir / "__init__.py"
    init.write_text("")
    result = module_to_utils_file(_UTILS, tmp_path)
    assert result == init


def test_module_file_takes_priority_over_init(tmp_path: Path) -> None:
    utils_dir = tmp_path / _UTILS
    utils_dir.mkdir()
    (utils_dir / "__init__.py").write_text("")
    sub = utils_dir / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    mod_file = utils_dir / "sub.py"
    mod_file.write_text("")
    result = module_to_utils_file(f"{_UTILS}.sub", tmp_path)
    assert result == mod_file


def test_neither_exists_returns_none(tmp_path: Path) -> None:
    assert module_to_utils_file(f"{_UTILS}.missing", tmp_path) is None


def test_collects_inits_along_path(tmp_path: Path) -> None:
    pkg = tmp_path / _UTILS
    pkg.mkdir()
    pkg_init = pkg / "__init__.py"
    pkg_init.write_text("")
    sub = pkg / "sub"
    sub.mkdir()
    sub_init = sub / "__init__.py"
    sub_init.write_text("")
    mod = sub / "module.py"
    mod.write_text("")

    inits = init_files_for_utils_file(mod, tmp_path)
    assert pkg_init in inits
    assert sub_init in inits


def test_missing_init_not_included(tmp_path: Path) -> None:
    pkg = tmp_path / _UTILS
    pkg.mkdir()
    sub = pkg / "sub"
    sub.mkdir()
    sub_init = sub / "__init__.py"
    sub_init.write_text("")
    mod = sub / "module.py"
    mod.write_text("")

    inits = init_files_for_utils_file(mod, tmp_path)
    assert sub_init in inits
    assert (pkg / "__init__.py") not in inits


def test_file_in_root_returns_empty(tmp_path: Path) -> None:
    mod = tmp_path / "module.py"
    mod.write_text("")
    assert init_files_for_utils_file(mod, tmp_path) == []


def test_no_utils_imports_empty_result(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text("import os\n")
    utils = tmp_path / "utils"
    utils.mkdir()
    assert collect_all_utils_files(app, utils) == set()


def test_direct_import_resolved(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text(f"from {_UTILS}.draw import x\n")

    utils = tmp_path / "utils"
    utils.mkdir()
    utils_pkg = utils / _UTILS
    utils_pkg.mkdir()
    draw_py = utils_pkg / "draw.py"
    draw_py.write_text("# no further imports\n")

    result = collect_all_utils_files(app, utils)
    assert draw_py in result


def test_transitive_imports_followed(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text(f"from {_UTILS}.draw import x\n")

    utils = tmp_path / "utils"
    utils.mkdir()
    utils_pkg = utils / _UTILS
    utils_pkg.mkdir()
    draw_py = utils_pkg / "draw.py"
    draw_py.write_text(f"from {_UTILS}.image_processing import y\n")
    img_py = utils_pkg / "image_processing.py"
    img_py.write_text("# leaf\n")

    result = collect_all_utils_files(app, utils)
    assert draw_py in result
    assert img_py in result


def test_unresolvable_module_warns_and_continues(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text(f"import {_UTILS}.missing_module\n")

    utils = tmp_path / "utils"
    utils.mkdir()
    (utils / _UTILS).mkdir()

    with pytest.warns(UserWarning, match="could not be resolved"):
        result = collect_all_utils_files(app, utils)
    assert result == set()
