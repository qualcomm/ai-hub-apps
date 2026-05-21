# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Bundle a Python app's source, shared SDK modules, and shell scripts.

The bundler:
  1. Verifies the app is a Python app.
  2. Scans all app .py files for imports from qai_hub_apps_utils.
  3. Copies only the needed SDK modules into out_dir (preserving the
     qai_hub_apps_utils/ directory structure so imports work unchanged).
  4. Reads the base SDK requirements.txt and requirements-<module>.txt for
     each copied SDK module, then merges with the app's requirements.txt
     into requirements.txt in out_dir.
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
from qai_hub_apps_test.bundlers.python.sdk_collector import (
    collect_all_sdk_files,
    init_files_for_sdk_file,
)
from qai_hub_apps_test.bundlers.python.sdk_resolver import resolve_sdk_root
from qai_hub_apps_test.bundlers.shell.bundle import bundle_scripts as _bundle_scripts
from qai_hub_apps_test.configs.info_yaml import AppLanguage, QAIHAAppInfo


def bundle_source(
    app_root: Path,
    out_dir: Path,
    sdk_parent: Path | None = None,
    shared_scripts_root: Path | None = None,
) -> None:
    """Copy app source, SDK modules, merged requirements, and shared scripts into out_dir.

    Parameters
    ----------
    app_root:
        Path to the Python app's root directory.
    out_dir:
        Destination directory (must not already exist).
    sdk_parent:
        Path to the directory containing ``qai_hub_apps_utils``. Auto-resolved
        from the repository structure if None.
    shared_scripts_root:
        Path to the shared shell scripts directory (``apps/_shared/scripts/``).
        Auto-resolved from the repository structure if None.
    """
    if sdk_parent is None:
        sdk_parent = resolve_sdk_root(None)
    app_info, _ = QAIHAAppInfo.from_app(app_root)

    if AppLanguage.PYTHON not in app_info.languages:
        raise ValueError(
            f"'{app_root}' is not a Python app "
            f"(languages={[l.value for l in app_info.languages]}). "
            "Expected 'Python' in languages."
        )

    app_id = app_info.id
    print(f"Bundling app '{app_id}' from {app_root}")

    # Collect needed SDK files
    sdk_files = collect_all_sdk_files(app_root, sdk_parent)
    if not sdk_files:
        print(
            "No qai_hub_apps_utils imports found; bundle will contain only app files."
        )
    else:
        print(f"Found {len(sdk_files)} SDK module file(s) to include.")

    # Collect requirements: base SDK requirements + per-module requirements files
    sdk_requires: list[str] = []
    sdk_base_req_file = sdk_parent / "requirements.txt"
    if sdk_base_req_file.exists():
        for line in sdk_base_req_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sdk_requires.append(line)
    for sdk_file in sdk_files:
        sdk_requires.extend(read_module_requirements(sdk_file))

    # Merge requirements
    app_req_file = app_root / "requirements.txt"
    merged_reqs = merge_requirements(app_req_file, sdk_requires)

    # Collect __init__.py files needed for SDK package structure
    all_sdk_files: set[Path] = set(sdk_files)
    for sdk_file in sdk_files:
        all_sdk_files.update(init_files_for_sdk_file(sdk_file, sdk_parent))

    shutil.copytree(app_root, out_dir)

    # SDK files
    for sdk_file in sorted(all_sdk_files):
        arcname = sdk_file.relative_to(sdk_parent)
        target = out_dir / arcname
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sdk_file, target)

    # requirements.txt
    bundle_reqs_content = "\n".join(merged_reqs) + "\n" if merged_reqs else ""
    (out_dir / "requirements.txt").write_text(bundle_reqs_content, encoding="utf-8")

    _bundle_scripts(out_dir, shared_scripts_root)
