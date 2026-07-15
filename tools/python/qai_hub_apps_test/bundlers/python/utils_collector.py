# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Collect qai_hub_apps_utils source files needed by an app."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

from qai_hub_apps_test.bundlers.python.utils_resolver import _UTILS_PACKAGE


def collect_utils_imports_from_file(py_file: Path) -> set[str]:
    """Return set of qai_hub_apps_utils module dotted names imported by py_file."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    except SyntaxError as e:
        warnings.warn(f"Could not parse {py_file}: {e}", stacklevel=2)
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == _UTILS_PACKAGE or name.startswith(_UTILS_PACKAGE + "."):
                    modules.add(name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == _UTILS_PACKAGE or mod.startswith(_UTILS_PACKAGE + "."):
                modules.add(mod)
    return modules


def module_to_utils_file(module: str, utils_parent: Path) -> Path | None:
    """
    Resolve a dotted module name to its .py file path under utils_parent.

    Tries two resolution strategies in order:
    1. ``<utils_parent>/<parts>.py`` — regular module file.
    2. ``<utils_parent>/<parts>/__init__.py`` — package init file.
    """
    parts = module.split(".")
    rel = Path(*parts)
    as_module = utils_parent / rel.with_suffix(".py")
    as_pkg = utils_parent / rel / "__init__.py"
    if as_module.exists():
        return as_module
    if as_pkg.exists():
        return as_pkg
    return None


def collect_all_utils_files(app_root: Path, utils_parent: Path) -> set[Path]:
    """
    Recursively find all qai_hub_apps_utils .py files needed by app_root sources.
    Also walks utils module files themselves for transitive imports.
    """
    py_files = list(app_root.rglob("*.py"))
    visited_modules: set[str] = set()
    needed_files: set[Path] = set()
    stack: list[str] = []

    # Seed from app source files
    for py_file in py_files:
        for mod in collect_utils_imports_from_file(py_file):
            if mod not in visited_modules:
                visited_modules.add(mod)
                stack.append(mod)

    # DFS over utils module imports
    while stack:
        mod = stack.pop()
        utils_file = module_to_utils_file(mod, utils_parent)
        if utils_file is None:
            warnings.warn(
                f"Imported qai_hub_apps_utils module '{mod}' could not be resolved "
                f"to a file under '{utils_parent}'. It will be skipped.",
                stacklevel=2,
            )
            continue
        if utils_file in needed_files:
            continue
        needed_files.add(utils_file)
        for transitive_mod in collect_utils_imports_from_file(utils_file):
            if transitive_mod not in visited_modules:
                visited_modules.add(transitive_mod)
                stack.append(transitive_mod)

    return needed_files


def init_files_for_utils_file(utils_file: Path, utils_parent: Path) -> list[Path]:
    """Return all __init__.py files from utils_parent down to utils_file's package."""
    inits: list[Path] = []
    rel = utils_file.relative_to(utils_parent)
    current = utils_parent
    for part in rel.parts[:-1]:  # exclude the file itself
        current = current / part
        init = current / "__init__.py"
        if init.exists():
            inits.append(init)
    return inits
