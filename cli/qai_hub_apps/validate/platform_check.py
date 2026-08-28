# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING

from qai_hub_apps.configs.app_yaml import AppType

if TYPE_CHECKING:
    from qai_hub_apps.registry import App


def check_platform(app: App) -> bool:
    """Return True if the app is supported on the current host platform."""
    app_type = app.app_type
    if app_type == AppType.ANDROID:
        return False

    platform_map = {AppType.WINDOWS: "win32", AppType.UBUNTU: "linux"}
    required_platform = platform_map.get(app_type)
    if required_platform is None:
        return True

    return sys.platform == required_platform


def get_host_info() -> str:
    return (
        f"OS: {platform.platform()}\n"
        f"System: {platform.system()}\n"
        f"OS release: {platform.release()}\n"
        f"OS version: {platform.version()}\n"
        f"Architecture: {platform.machine()}\n"
        f"Processor: {platform.processor()}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Python executable: {sys.executable}\n"
    )
