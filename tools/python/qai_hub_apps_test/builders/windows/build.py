# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Build Windows C++ apps."""

from __future__ import annotations

from pathlib import Path


def build_cpp_app(app_dir: Path) -> None:
    """Build Windows C++ app.

    Parameters
    ----------
    app_dir:
        Root directory of the fetched Windows C++ app.
    """
    raise NotImplementedError("Windows C++ app builds are not yet implemented.")
