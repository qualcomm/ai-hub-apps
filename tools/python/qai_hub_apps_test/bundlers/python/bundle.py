# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Bundle a Python app's source, shared qai_hub_apps_utils modules, and shell scripts.

The bundler:
  1. Verifies the app is a Python app.
  2. Scans all app .py files for imports from qai_hub_apps_utils.
  3. Copies only the needed qai_hub_apps_utils modules into out_dir (preserving
     the qai_hub_apps_utils/ directory structure so imports work unchanged).
  4. Reads the base qai_hub_apps_utils requirements.txt and
     requirements-<module>.txt for each copied utils module, then merges with
     the app's requirements.txt into requirements.txt in out_dir.
  5. Copies referenced shared shell scripts into out_dir/scripts/ and
     rewrites source/dot-source lines to bundle-local paths.

Orchestration (temp dir creation, zip/copy finalization)
is handled by bundle_app() in bundlers/__init__.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from qai_hub_apps_test.bundlers.python.requirements import (
    merge_requirements,
    read_module_requirements,
)
from qai_hub_apps_test.bundlers.python.utils_collector import (
    collect_all_utils_files,
    init_files_for_utils_file,
)
from qai_hub_apps_test.bundlers.python.utils_resolver import resolve_utils_root
from qai_hub_apps_test.bundlers.shell.bundle import bundle_scripts as _bundle_scripts
from qai_hub_apps_test.configs.info_yaml import AppLanguage, QAIHAAppInfo


def bundle_source(
    app_root: Path,
    out_dir: Path,
    utils_parent: Path | None = None,
    shared_scripts_root: Path | None = None,
) -> None:
    """Copy app source, utils modules, merged requirements, and shared scripts into out_dir.

    Parameters
    ----------
    app_root:
        Path to the Python app's root directory.
    out_dir:
        Destination directory (must not already exist).
    utils_parent:
        Path to the directory containing ``qai_hub_apps_utils``. Auto-resolved
        from the repository structure if None.
    shared_scripts_root:
        Path to the shared shell scripts directory (``apps/_shared/scripts/``).
        Auto-resolved from the repository structure if None.
    """
    if utils_parent is None:
        utils_parent = resolve_utils_root(None)
    app_info, _ = QAIHAAppInfo.from_app(app_root)

    if AppLanguage.PYTHON not in app_info.languages:
        raise ValueError(
            f"'{app_root}' is not a Python app "
            f"(languages={[l.value for l in app_info.languages]}). "
            "Expected 'Python' in languages."
        )

    app_id = app_info.id
    print(f"Bundling app '{app_id}' from {app_root}")

    # Collect needed qai_hub_apps_utils files
    utils_files = collect_all_utils_files(app_root, utils_parent)
    if not utils_files:
        print(
            "No qai_hub_apps_utils imports found; bundle will contain only app files."
        )
    else:
        print(f"Found {len(utils_files)} qai_hub_apps_utils module file(s) to include.")

    # Collect requirements: base utils requirements + per-module requirements files
    utils_requires: list[str] = []
    utils_base_req_file = utils_parent / "requirements.txt"
    if utils_base_req_file.exists():
        for line in utils_base_req_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                utils_requires.append(line)
    for utils_file in utils_files:
        utils_requires.extend(read_module_requirements(utils_file))

    # Merge requirements
    app_req_file = app_root / "requirements.txt"
    merged_reqs = merge_requirements(app_req_file, utils_requires)

    # Collect __init__.py files needed for qai_hub_apps_utils package structure
    all_utils_files: set[Path] = set(utils_files)
    for utils_file in utils_files:
        all_utils_files.update(init_files_for_utils_file(utils_file, utils_parent))

    shutil.copytree(app_root, out_dir)

    # qai_hub_apps_utils files
    for utils_file in sorted(all_utils_files):
        arcname = utils_file.relative_to(utils_parent)
        target = out_dir / arcname
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(utils_file, target)

    # requirements.txt
    bundle_reqs_content = "\n".join(merged_reqs) + "\n" if merged_reqs else ""
    (out_dir / "requirements.txt").write_text(bundle_reqs_content, encoding="utf-8")

    _bundle_scripts(out_dir, shared_scripts_root)
