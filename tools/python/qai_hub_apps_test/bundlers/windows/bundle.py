# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""Bundle a Windows C++ app's source and shared PowerShell scripts.

Windows C++ apps (e.g. chatapp_windows_cpp) ship Visual Studio project files
and PowerShell install/test scripts. Bundling copies the app source as-is and
copies the referenced shared scripts into ``scripts/``, rewriting the
``. ..\_shared\scripts\foo.ps1`` dot-source lines to bundle-local paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from qai_hub_apps_test.bundlers.shell.bundle import bundle_scripts as _bundle_scripts


def bundle_source(
    app_root: Path,
    out_dir: Path,
    shared_scripts_root: Path | None = None,
) -> None:
    """Copy app source and shared scripts into out_dir.

    Parameters
    ----------
    app_root:
        Path to the Windows C++ app's root directory.
    out_dir:
        Destination directory (must not already exist).
    shared_scripts_root:
        Path to the shared shell scripts directory (``apps/_shared/scripts/``).
        Auto-resolved from the repository structure if None.
    """
    shutil.copytree(app_root, out_dir)
    _bundle_scripts(out_dir, shared_scripts_root)
