# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

from qai_hub_apps_test.builders.android.build import build_app as _build_android_app
from qai_hub_apps_test.configs.info_yaml import AppType, QAIHAAppInfo


def build_app(app_info: QAIHAAppInfo, app_dir: Path) -> None:
    """Build the fetched app. No-op for app types that need no build step.

    Parameters
    ----------
    app_info:
        App metadata.
    app_dir:
        Root directory of the fetched app.
    """
    if app_info.app_type == AppType.ANDROID:
        _build_android_app(app_dir)
    elif app_info.app_type == AppType.UBUNTU:
        pass  # Ubuntu Python apps: nothing to build
    else:
        raise NotImplementedError(
            f"Build not implemented for app_type={app_info.app_type.value}"
        )
