# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Resolve the qai_hub_apps_utils root directory."""

from __future__ import annotations

import sys
from pathlib import Path

from qai_hub_apps_test.utils.paths import REPOSITORY_ROOT

_UTILS_PACKAGE = "qai_hub_apps_utils"
_DEFAULT_UTILS_PARENT = REPOSITORY_ROOT / "apps" / "_shared" / "python"


def resolve_utils_root(utils_root_arg: str | None) -> Path:
    if utils_root_arg:
        p = Path(utils_root_arg).resolve()
        # Accept either the package dir itself or its parent
        if (p / "__init__.py").exists() and p.name == _UTILS_PACKAGE:
            return p.parent
        if (p / _UTILS_PACKAGE / "__init__.py").exists():
            return p
        sys.exit(
            f"error: --utils_root '{p}' does not contain a '{_UTILS_PACKAGE}' package."
        )
    if not (_DEFAULT_UTILS_PARENT / _UTILS_PACKAGE / "__init__.py").exists():
        sys.exit(
            f"error: qai_hub_apps_utils not found at default location "
            f"'{_DEFAULT_UTILS_PARENT}'. Pass --utils_root pointing to the directory "
            "that contains the qai_hub_apps_utils/ package."
        )
    return _DEFAULT_UTILS_PARENT
