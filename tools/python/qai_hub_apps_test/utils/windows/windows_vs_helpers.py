# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import subprocess
from pathlib import Path

from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.device import ScorecardDevice

from qai_hub_apps_test.configs.info_yaml import QAIHAAppInfo
from qai_hub_apps_test.configs.versions_yaml import VersionsRegistry
from qai_hub_apps_test.utils.models.install_model import install_model
from qai_hub_apps_test.utils.verify_result import VerifyResult


def verify_windows_app_versions_match(
    app_root: str | os.PathLike,
    app_info: QAIHAAppInfo,
    versions: VersionsRegistry | None = None,
) -> VerifyResult:
    if versions is None:
        versions = VersionsRegistry.load()
    """
    Verifies that the Windows app at the given root directory
    uses the same versions of runtimes as are listed in the versions yaml.

    Parameters
    ----------
    app_root
        Path to the root directory of the application source code.
    versions
        Registry of version information for dependencies and components.

    Returns
    -------
    VerifyResult
        The warnings and errors produced by this verification process.
    """
    errors: list[str] = []
    warnings: list[str] = []
    # TODO: check version for VS build tools?
    return VerifyResult(errors, warnings)


def build_windows_app(
    app_info: QAIHAAppInfo,
    app_root: str | os.PathLike,
    model_id: str | None = None,
    precision: Precision | None = None,
    device: ScorecardDevice | None = None,
    qaihm_version_tag: str | None = None,
    clean_build: bool = False,
) -> None:
    """
    Builds a Windows application using the specified configuration and model parameters.

    Parameters
    ----------
    app_info
        Metadata and configuration details for the app to be built.
    app_root
        Path to the root directory of the application source code.
    model_id
        AI Hub Models ID for the model to be embedded in the app. If None, any app-supported model is included.
    precision
        Desired precision level for the model (e.g., FP32, INT8). If None, any supported precision for the app may be used.
    device
        Target device for fetching model assets. If None, assets used by this app must be universal.
    qaihm_version_tag
        Version tag of AI Hub Models from which models should be fetched. If None, the currently installed version is used.
    clean_build
        If True, performs a clean build by removing previous build artifacts.
    """
    app_root = Path(app_root)
    model_id = model_id or app_info.related_models[0]
    precision = precision or app_info.precisions[0]

    # Note: msbuild is only recognizable inside a VS Dev command prompt or load VSDevCmd.bat before building/cleaning
    assert os.getenv("VSDEV_CMD"), (
        "Couldn't find VS Dev Command prompt, $VSDEV_CMD should be set to path of VSDevCmd.bat batch file"
    )

    # Build the app
    if clean_build:
        clean_windows_app(app_root)
    install_model(
        app_root,
        app_info,
        app_info.model_file_paths,
        model_id,
        app_info.runtime,
        precision,
        device,
        qaihm_version_tag,
    )

    subprocess.run(
        f'"{os.environ["VSDEV_CMD"]}" && msbuild -t:restore -p:RestorePackagesConfig=true',
        cwd=app_root,
        text=True,
        shell=True,
        check=True,
    )
    subprocess.run(
        f'"{os.environ["VSDEV_CMD"]}" && vcpkg integrate install',
        cwd=app_root,
        text=True,
        shell=True,
        check=True,
    )
    subprocess.run(
        f'"{os.environ["VSDEV_CMD"]}" && msbuild ',
        cwd=app_root,
        text=True,
        shell=True,
        check=True,
    )


def clean_windows_app(
    app_root: str | os.PathLike,
) -> None:
    # TODO: add custom Target for cleanup through msbuild
    for dir_ in ["ARM64", "packages", "vcpkg_installed", os.path.basename(app_root)]:
        subprocess.run(
            f'if exist "{app_root}\\{dir_}" rd /s /q "{app_root}\\{dir_}"',
            cwd=app_root,
            text=True,
            shell=True,
            check=True,
        )
