# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from qai_hub_apps.configs.app_yaml import AppType
from qai_hub_apps.errors import AppIncompatibleError, QAIHubAppsError

if TYPE_CHECKING:
    from qai_hub_apps.registry import App

logger = logging.getLogger(__name__)


def ensure_docker_available() -> None:
    """Raise a clear error if Docker is not installed or the daemon is unreachable."""
    logger.debug("Checking Docker availability")
    if shutil.which("docker") is None:
        raise QAIHubAppsError(
            "Docker is required for this build but was not found on PATH.\n"
            "Install Docker (https://docs.docker.com/get-started/get-docker/) or "
            "re-run with --no-docker to build natively."
        )
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        raise QAIHubAppsError(
            "Docker is installed but the daemon is not reachable. "
            "Start Docker and try again, or re-run with --no-docker."
        ) from e
    logger.debug("Docker is available and the daemon is reachable")


def ensure_build_supported(app: App, use_docker: bool) -> None:
    """Raise ``AppIncompatibleError`` if the app cannot be built on this host.

    Validates the host/mode combination before the build script runs:

    - Android apps can only be built on Linux or under WSL.
    - Windows apps can only be built on a Windows host.
    - Windows C++ Docker builds use Windows container images, which cannot be
      built on an ARM64 Docker daemon; build natively (``--no-docker``) or use an
      x86-64 Docker host.
    - Docker-mode builds require Docker to be installed and running.

    Ubuntu / Windows-Python apps have no build-time host constraint (their build
    is a no-op).
    """
    # platform.machine() reports the process arch, note that an emulated x64 Python on
    # an ARM64 Windows host reads as AMD64.
    machine = platform.machine()
    logger.debug(
        "Checking build support for '%s': app_type=%s, use_docker=%s, "
        "platform=%s, machine=%s",
        app.id,
        app.app_type.value,
        use_docker,
        sys.platform,
        machine,
    )
    if app.app_type == AppType.ANDROID and sys.platform == "win32":
        raise AppIncompatibleError(
            f"'{app.id}' is an Android app and can only be built on Linux or under "
            "WSL (https://learn.microsoft.com/windows/wsl/install)."
        )
    if app.app_type == AppType.WINDOWS and sys.platform != "win32":
        raise AppIncompatibleError(
            f"'{app.id}' is a Windows app and can only be built on Windows "
            f"(detected platform: {sys.platform})."
        )
    if (
        app.app_type == AppType.WINDOWS
        and use_docker
        and machine.lower() in ("arm64", "aarch64")
    ):
        raise AppIncompatibleError(
            f"'{app.id}' Docker build uses a Windows container image, which "
            "cannot be built on an ARM64 host. Build natively with --no-docker, "
            "or use an x86-64 Docker host."
        )
    if use_docker:
        ensure_docker_available()
    logger.debug("Build support check passed for '%s'", app.id)
