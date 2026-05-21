# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Bundle Android app source into a standalone directory.

Android apps share code via symlinks (e.g. tflite helpers, common.gradle).
Bundling resolves all symlinks so the output is fully self-contained.
Version variables in build.gradle are inlined with their resolved values
loaded from versions.env copied into the bundle by the shell bundler.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from qai_hub_apps_test.bundlers.shell.bundle import bundle_scripts as _bundle_scripts
from qai_hub_apps_test.utils.versions import load_versions


def _gradle_literal(value: str) -> str:
    """Return value as a Gradle literal, bare int or single-quoted string."""
    return value if value.isdigit() else f"'{value}'"


def _inline_versions(build_gradle: Path, versions: dict[str, str]) -> None:
    """Replace ${VAR} and bare VAR references in build.gradle with resolved values.

    Also empties common.gradle so the apply from succeeds but does nothing
    (versions are now inlined; common.gradle is not needed in a bundle).
    """
    content = build_gradle.read_text(encoding="utf-8")
    original = content

    # Handle references inside double-quoted strings, like:
    # api "org.tensorflow:tensorflow-lite:${TF_LITE_VERSION}"
    content = re.sub(
        r"\$\{([A-Z_][A-Z0-9_]*)\}",
        lambda m: versions.get(m.group(1), m.group(0)),
        content,
    )

    # Handle references that appear as bare identifiers, like:
    # ndkVersion ANDROID_NDK_VERSION
    for key, value in versions.items():
        content = re.sub(
            rf'(?<!["\'\w]){re.escape(key)}(?!["\'\w])',
            _gradle_literal(value),
            content,
        )

    inlined = sum(1 for k in versions if k in original)
    print(f"Inlined {inlined} version variable(s) into {build_gradle.name}.")

    build_gradle.write_text(content, encoding="utf-8")

    # clear common.gradle content so that 'apply from...' resolves
    common_gradle = build_gradle.parent / "_shared" / "android" / "common.gradle"
    if common_gradle.exists():
        common_gradle.write_text("")
        print(f"Cleared {common_gradle.name}.")


def bundle_source(
    app_root: Path,
    out_dir: Path,
    shared_scripts_root: Path | None = None,
) -> None:
    """Copy app source to out_dir, resolving all symlinks, then inline versions.

    Parameters
    ----------
    app_root:
        Path to the Android app's root directory.
    out_dir:
        Destination directory (must not already exist).
    shared_scripts_root:
        Path to the shared shell scripts directory (``apps/_shared/scripts/``).
        Auto-resolved from the repository structure if None.
    """
    shutil.copytree(app_root, out_dir, symlinks=False)
    _bundle_scripts(out_dir, shared_scripts_root)
    versions = load_versions(out_dir / "scripts" / "versions.env")
    _inline_versions(out_dir / "build.gradle", versions)
