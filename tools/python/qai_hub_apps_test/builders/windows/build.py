# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Build Windows C++ apps, either via Docker or natively on a Windows host."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tap import Tap
from tenacity import retry, stop_after_attempt, wait_fixed


class WindowsBuilderParser(Tap):
    app_dir: Path
    use_docker: bool = False


def _resolve_sln(app_dir: Path) -> str:
    """Return the single ``.sln`` file name in ``app_dir``."""
    sln_files = list(app_dir.glob("*.sln"))
    if len(sln_files) != 1:
        raise ValueError(
            f"Expected exactly one '.sln' in '{app_dir}', found {len(sln_files)}: "
            f"{[s.name for s in sln_files]}"
        )
    return sln_files[0].name


@retry(reraise=True, wait=wait_fixed(30), stop=stop_after_attempt(2))
def _build_cpp_docker(app_dir: Path, sln_name: str) -> None:
    r"""Build inside a Windows Docker container, then copy ``ARM64\`` back to the host.

    Windows container images cannot be built on a Linux daemon, so this requires
    a Windows Docker host.
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
    no_cache = (
        ["--no-cache"] if _build_cpp_docker.statistics["attempt_number"] > 1 else []
    )
    subprocess.run(
        [
            "docker",
            "build",
            *no_cache,
            "--build-arg",
            "BUILD_TYPE=build",
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
                "powershell",
                "-Command",
                ". ./install_build.ps1; "
                f"& $env:MSBUILD_EXE {sln_name} /p:Configuration=Release /p:Platform=ARM64; "
                "exit $LASTEXITCODE",
            ],
            check=True,
        )

        # Copy the build output back to the host
        subprocess.run(
            ["docker", "cp", f"{container_name}:C:\\app\\ARM64", str(app_dir)],
            check=True,
        )
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], check=False)
        subprocess.run(["docker", "rmi", image_tag], check=False)


def _build_cpp_native(app_dir: Path, sln_name: str) -> None:
    r"""Build directly on the Windows host (no Docker).

    Runs ``install_build.ps1`` then MSBuild, in ``app_dir``. The ``ARM64\`` output lands in ``app_dir`` directly.
    """
    subprocess.run(
        [
            "powershell",
            "-Command",
            ". ./install_build.ps1; "
            f"& $env:MSBUILD_EXE {sln_name} /p:Configuration=Release /p:Platform=ARM64; "
            "exit $LASTEXITCODE",
        ],
        cwd=app_dir,
        check=True,
    )


def build_cpp_app(app_dir: Path, use_docker: bool = True) -> None:
    """Build a Windows C++ app's ARM64 binaries.

    Parameters
    ----------
    app_dir:
        Root directory of the fetched Windows C++ app (must contain a ``.sln``,
        ``install_build.ps1``, ``scripts/`` from the bundle, and — for Docker
        builds — a Dockerfile).
    use_docker:
        If True, build inside a Windows Docker container; otherwise build
        natively on the host.
    """
    sln_name = _resolve_sln(app_dir)
    if use_docker:
        _build_cpp_docker(app_dir, sln_name)
    else:
        _build_cpp_native(app_dir, sln_name)


if __name__ == "__main__":
    args = WindowsBuilderParser().parse_args()
    build_cpp_app(args.app_dir, args.use_docker)
