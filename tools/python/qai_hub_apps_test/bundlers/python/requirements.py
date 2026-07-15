# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Extract and merge requirements for a bundled Python app."""

from __future__ import annotations

import re
import warnings
from pathlib import Path


def read_module_requirements(py_file: Path) -> list[str]:
    """Read requirements for a qai_hub_apps_utils module.

    Looks for requirements-<module>.txt in the requirements/ subfolder
    next to the module file (e.g. draw.py -> requirements/requirements-draw.txt).
    """
    req_file = py_file.parent / "requirements" / f"requirements-{py_file.stem}.txt"
    if not req_file.exists():
        warnings.warn(
            f"No requirements file found for qai_hub_apps_utils module '{py_file.name}' "
            f"(expected {req_file}). No dependencies will be added for this module.",
            stacklevel=2,
        )
        return []
    return [
        line.strip()
        for line in req_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _pkg_name(req: str) -> str:
    """Return the package name part of a requirement string (before any specifier)."""
    for sep in ("==", ">=", "<=", "!=", "~=", ">", "<", "["):
        req = req.split(sep)[0]
    return re.sub(r"[-_.]+", "-", req.strip().lower())


def merge_requirements(app_req_file: Path, utils_requires: list[str]) -> list[str]:
    app_reqs: list[str] = []
    if app_req_file.exists():
        for line in app_req_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                app_reqs.append(line)

    combined: dict[str, list[str]] = {}
    for req in app_reqs + utils_requires:
        name = _pkg_name(req)
        combined.setdefault(name, [])
        if req not in combined[name]:
            combined[name].append(req)

    result: list[str] = []
    for name, entries in combined.items():
        if len(entries) > 1:
            warnings.warn(
                f"Package '{name}' appears with multiple specifiers: {entries}. "
                "All entries are included in requirements.txt.",
                stacklevel=2,
            )
        result.extend(entries)

    return result
