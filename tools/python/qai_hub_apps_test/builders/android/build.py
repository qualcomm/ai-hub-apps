# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Build Android APKs for a fetched app, either via Docker or natively."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_fixed


@retry(reraise=True, wait=wait_fixed(30), stop=stop_after_attempt(2))
def _build_docker(app_dir: Path) -> None:
    """Build the debug + test APKs inside a Docker container, then copy outputs back.

    Builds a Docker image with BUILD_TYPE=build (which runs install_build.sh), runs gradle assembleDebug assembleAndroidTest inside
    the container, then copies build/outputs/ back to app_dir via docker cp.
    """
    if not (app_dir / "Dockerfile").is_file():
        raise FileNotFoundError(
            f"No 'Dockerfile' found in '{app_dir}'. "
            "Ensure the app declares 'base_docker' in info.yaml and was bundled "
            "with bundle_app() before building."
        )

    image_tag = f"aiha-build-{app_dir.name}"
    container_name = f"aiha-build-container-{app_dir.name}"

    # A corrupt/exhausted build cache is a common cause of a transient build
    # failure, so on a retry rebuild this image with --no-cache.
    no_cache = ["--no-cache"] if _build_docker.statistics["attempt_number"] > 1 else []
    subprocess.run(
        [
            "docker",
            "build",
            *no_cache,
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


def _build_native(app_dir: Path) -> None:
    raise NotImplementedError("Native building of android apps is not supported")


def build_app(app_dir: Path, use_docker: bool = True) -> None:
    """Build Android debug and test APKs.

    Parameters
    ----------
    app_dir:
        Root directory of the fetched Android app (must contain install_build.sh,
        scripts/ from the bundle, and — for Docker builds — a Dockerfile).
    use_docker:
        If True, build inside a Docker container; otherwise build natively on the host.
    """
    if use_docker:
        _build_docker(app_dir)
    else:
        _build_native(app_dir)
