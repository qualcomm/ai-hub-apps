# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from enum import Enum, unique
from pathlib import Path

from pydantic import ConfigDict, Field, field_validator, model_validator
from qai_hub_models.configs.info_yaml import MODEL_LICENSE as LICENSE
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_8_gen_3, cs_x_elite
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from typing_extensions import assert_never

from qai_hub_apps_test.utils.paths import APPS_ROOT, REPOSITORY_ROOT


@unique
class AppStatus(Enum):
    UNPUBLISHED = "unpublished"
    PUBLISHED = "published"


@unique
class AppLanguage(Enum):
    PYTHON = "Python"
    CPP = "C++"
    JAVA = "Java"
    KOTLIN = "Kotlin"


@unique
class AppType(Enum):
    ANDROID = "android"
    WINDOWS = "windows"
    UBUNTU = "ubuntu"

    @property
    def default_device(self) -> ScorecardDevice:
        if self == AppType.ANDROID:
            return cs_8_gen_3
        if self == AppType.WINDOWS:
            return cs_x_elite
        if self == AppType.UBUNTU:
            return cs_x_elite  # safe-harbor; no device has ubuntu os yet
        assert_never(self)


class AppUrl(BaseQAIHMConfig):
    source: str


class QAIHACLIAppInfo(BaseQAIHMConfig):
    """CLI-facing subset of app info — the fields written to registry.yaml."""

    model_config = ConfigDict(extra="ignore")

    name: str
    id: str
    status: AppStatus
    headline: str
    description: str
    domain: str
    use_case: str
    app_repo_url: str | None = None
    app_type: AppType
    runtime: TargetRuntime
    related_models: list[str]
    precisions: list[Precision]
    model_file_paths: list[
        Path
    ] = []  # Destination paths for each downloaded model file
    model_file_dir: str | None = None  # Directory to unzip all model files into
    url: AppUrl | None = None

    @model_validator(mode="after")
    def _validate_model_location(self) -> "QAIHACLIAppInfo":
        has_paths = bool(self.model_file_paths)
        has_dir = self.model_file_dir is not None
        if has_paths and has_dir:
            raise ValueError(
                "model_file_paths and model_file_dir are mutually exclusive; set only one."
            )
        if has_paths and len(self.model_file_paths) > 1:
            parents = {Path(p).parent for p in self.model_file_paths}
            if len(parents) > 1:
                raise ValueError(
                    f"All model_file_paths must share the same parent directory, "
                    f"got: {sorted(str(p) for p in parents)}"
                )
        return self


class QAIHAAppInfo(QAIHACLIAppInfo):
    """Full internal app info — adds CI/build fields on top of CLI-facing fields."""

    model_config = ConfigDict(extra="ignore")

    ##########################
    # General Information
    ##########################

    skip_test: str | None = None
    include_in_cli: bool = True
    supported_devices: list[ScorecardDevice] = Field(default_factory=list)
    app_repo_relative_path: str | None = (
        None  # relative path within qualcomm/ai-hub-apps
    )

    @field_validator("supported_devices", mode="before")
    @classmethod
    def _parse_devices(cls, v: list) -> list[ScorecardDevice]:
        return [ScorecardDevice.parse(d) if isinstance(d, str) else d for d in v]

    @model_validator(mode="after")
    def _validate_repo(self) -> "QAIHAAppInfo":
        if not self.app_repo_url and not self.app_repo_relative_path:
            raise ValueError(
                f"App '{self.id}': one of app_repo_url or app_repo_relative_path must be set"
            )
        return self

    # License information
    license_url: str
    license_type: LICENSE

    languages: list[AppLanguage]

    ##########################
    # Build System Information
    ##########################

    # Supported AI Hub Models version
    # If None, assumes any version is supported.
    qaihm_version: str | None = None

    # Path to private S3 URLs that CI will use to fetch certain models. map<Model ID, map<Precision, map<Chipset, Relative S3 Path>>
    # This is necessary for complex models (like LLMs) until AI Hub Models has a good way to fetch these.
    private_model_s3_paths: dict[str, dict[Precision, dict[str, str]]] = Field(
        default_factory=dict
    )

    @staticmethod
    def from_app(path: str | os.PathLike) -> tuple["QAIHAAppInfo", Path]:
        """
        Load an app info from this directory or yaml file.

        If the path is relative, dir is assumed to be relative to qai-hub-apps/apps.
        """
        path = Path(path)
        if not os.path.isabs(path):
            if path.parts and path.parts[0] == "apps":
                path = REPOSITORY_ROOT / path
            else:
                path = APPS_ROOT / path
        yaml_path = path / "info.yaml" if os.path.isdir(path) else path
        adir = path if os.path.isdir(path) else Path(os.path.dirname(path))
        return QAIHAAppInfo.from_yaml(yaml_path), adir
