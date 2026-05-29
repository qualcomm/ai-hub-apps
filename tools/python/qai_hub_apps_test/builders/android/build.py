# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Build Android APKs for a fetched app using Docker."""

from __future__ import annotations

import subprocess
from pathlib import Path


def build_app(app_dir: Path) -> None:
    """Build Android debug and test APKs using Docker.

    Expects a ``Dockerfile`` to already exist in ``app_dir`` (injected by
    ``bundle_app()`` from the app's ``base_docker`` field). Builds a Docker
    image with BUILD_TYPE=build (which runs install_build.sh to install the
    Android SDK), runs gradle assembleDebug assembleAndroidTest inside the
    container, then copies build/outputs/ back to app_dir on the host via
    docker cp.

    Parameters
    ----------
    app_dir:
        Root directory of the fetched Android app (must contain install_build.sh,
        scripts/ from the bundle, and a Dockerfile).
    """
    if not (app_dir / "Dockerfile").is_file():
        raise FileNotFoundError(
            f"No 'Dockerfile' found in '{app_dir}'. "
            "Ensure the app declares 'base_docker' in info.yaml and was bundled "
            "with bundle_app() before building."
        )

    image_tag = f"aiha-build-{app_dir.name}"
    container_name = f"aiha-build-container-{app_dir.name}"

    subprocess.run(
        [
            "docker",
            "build",
            "--build-arg",
            "BUILD_TYPE=build",
            "--build-arg",
            "REGISTRY_PREFIX=docker-registry.qualcomm.com/library/",
            "--build-arg",
            "INSTALL_QUALCOMM_CA=true",
            "-t",
            image_tag,
            ".",
        ],
        cwd=app_dir,
        check=True,
    )
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--name",
                container_name,
                image_tag,
                "bash",
                "-c",
                "source /app/scripts/android_utils.sh && "
                "cd /app && gradle assembleDebug assembleAndroidTest",
            ],
            check=True,
        )

        # Copy APKs back to host
        outputs_dir = app_dir / "build" / "outputs"
        outputs_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "docker",
                "cp",
                f"{container_name}:/app/build/outputs",
                str(outputs_dir.parent),
            ],
            check=True,
        )
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], check=False)
        subprocess.run(["docker", "rmi", image_tag], check=False)
