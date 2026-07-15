# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from qai_hub_apps_test.bundlers.android.bundle import (
    bundle_source as _bundle_android_source,
)
from qai_hub_apps_test.bundlers.python.bundle import (
    bundle_source as _bundle_python_source,
)
from qai_hub_apps_test.bundlers.windows.bundle import (
    bundle_source as _bundle_windows_cpp_source,
)
from qai_hub_apps_test.configs.info_yaml import AppLanguage, AppType, QAIHAAppInfo
from qai_hub_apps_test.utils.paths import DOCKER_ROOT, find_app_dir


def bundle_app(
    app: str | Path,
    output_dir: Path,
    utils_parent: Path | None = None,
    shared_scripts_root: Path | None = None,
    make_zip: bool = False,
) -> None:
    """Bundle an app by app ID or directory path.

    - **Android** apps: deep-copy resolving all symlinks, copy shared scripts,
      then inline ``ext`` version variables into ``build.gradle`` and empty
      ``common.gradle``.
    - **Python** apps: copy source + shared qai_hub_apps_utils modules + merged
      ``requirements.txt``, then copy and rewrite shared shell scripts.

    All variants stage into a temporary directory, then either copy to
    ``output_dir/<app_id>/`` or zip to ``output_dir/<app_id>.zip``.

    Parameters
    ----------
    app:
        Either a string app ID (resolved via find_app_dir) or a Path to
        the app's root directory.
    output_dir:
        Directory where the bundle will be written.
    utils_parent:
        Path to the directory containing ``qai_hub_apps_utils``. Auto-resolved
        from the repository structure if None. (Python apps only.)
    shared_scripts_root:
        Path to the shared shell scripts directory (``apps/_shared/scripts/``).
        Auto-resolved from the repository structure if None.
    make_zip:
        If True, produce a zip archive; otherwise copy to a subdirectory.

    Raises
    ------
    NotImplementedError
        If the app type/language combination is not supported for bundling.
    """
    app_dir: Path = find_app_dir(app) if isinstance(app, str) else app
    app_info, _ = QAIHAAppInfo.from_app(app_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp) / app_info.id

        if app_info.app_type == AppType.ANDROID:
            _bundle_android_source(app_dir, tmp_dir, shared_scripts_root)
        elif AppLanguage.PYTHON in app_info.languages:
            _bundle_python_source(app_dir, tmp_dir, utils_parent, shared_scripts_root)
        elif (
            app_info.app_type == AppType.WINDOWS
            and AppLanguage.CPP in app_info.languages
        ):
            _bundle_windows_cpp_source(app_dir, tmp_dir, shared_scripts_root)
        else:
            raise NotImplementedError(
                f"App '{app_info.id}' (type={app_info.app_type.value}, "
                f"languages={[lang.value for lang in app_info.languages]}) "
                "is not supported for bundling."
            )

        if app_info.base_docker is not None:
            src_dockerfile = DOCKER_ROOT / app_info.base_docker
            if not src_dockerfile.is_file():
                raise FileNotFoundError(
                    f"Dockerfile '{app_info.base_docker}' not found at '{src_dockerfile}'. "
                    "Check the base_docker field in info.yaml."
                )
            shutil.copy2(src_dockerfile, tmp_dir / "Dockerfile")

        if make_zip:
            zip_path = output_dir / f"{app_info.id}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(tmp_dir.rglob("*")):
                    if f.is_file():
                        zf.write(f, f.relative_to(tmp_dir))
            print(f"Bundle written to: {zip_path}")
        else:
            dest = output_dir / app_info.id
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(tmp_dir, dest)
            print(f"Bundle written to: {dest}")
