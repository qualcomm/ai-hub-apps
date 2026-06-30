# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from enum import Enum, unique
from pathlib import Path

from pydantic import ConfigDict, Field, field_validator, model_validator

from qai_hub_apps_test.configs.base_config import BaseConfig
from qai_hub_apps_test.configs.field_types import Device, Precision, TargetRuntime
from qai_hub_apps_test.utils.paths import APPS_ROOT, REPOSITORY_ROOT


@unique
class AppStatus(Enum):
    UNPUBLISHED = "unpublished"  # WIP: not on the website or CLI, but tested in CI
    PUBLISHED = "published"  # Live on the website, in the CLI, and tested
    PUBLISHED_WEBSITE_ONLY = (
        "published_website_only"  # On the website only; not CLI-fetchable or tested
    )
    DEPRECATED = "deprecated"  # Like published, but carries a deprecation notice


@unique
class AppLicense(Enum):
    UNLICENSED = "unlicensed"
    COMMERCIAL = "commercial"
    APACHE_2_0 = "apache-2.0"
    MIT = "mit"
    BSD_3_CLAUSE = "bsd-3-clause"
    CC_BY_4_0 = "cc-by-4.0"
    AGPL_3_0 = "agpl-3.0"
    GPL_3_0 = "gpl-3.0"
    OTHER_NON_COMMERCIAL = "other-non-commercial"
    CC_BY_NON_COMMERCIAL_4_0 = "cc-by-non-commercial-4.0"


@unique
class AppLanguage(Enum):
    PYTHON = "Python"
    CPP = "C++"
    JAVA = "Java"
    KOTLIN = "Kotlin"
    GO = "Go"


@unique
class AppType(Enum):
    ANDROID = "android"
    WINDOWS = "windows"
    UBUNTU = "ubuntu"


class AppUrl(BaseConfig):
    source: str


class QAIHACLIAppInfo(BaseConfig):
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
    runtime: list[TargetRuntime]
    related_models: list[str]
    precisions: list[Precision]
    languages: list[AppLanguage]
    model_file_paths: list[
        Path
    ] = []  # Destination paths for each downloaded model file
    model_file_dir: str | None = None  # Directory to unzip all model files into
    # Set True for apps that download their model at runtime
    disable_cli_model_fetch: bool = False
    url: AppUrl | None = None
    # Optional message shown by the CLI for DEPRECATED apps. If unset, a default
    # deprecation message is used.
    deprecation_notice: str | None = None

    # Supported AI Hub Models version (this version is used by the CLI to download model assets)
    # If None, assumes any version is supported.
    qaihm_version: str | None = None

    @field_validator("runtime", mode="before")
    @classmethod
    def _normalize_runtime(cls, value: object) -> object:
        """Normalize runtime; a single value is wrapped into a list."""
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def _validate_model_location(self) -> "QAIHACLIAppInfo":
        has_paths = bool(self.model_file_paths)
        has_dir = self.model_file_dir is not None
        if has_paths and has_dir:
            raise ValueError(
                "model_file_paths and model_file_dir are mutually exclusive; set only one."
            )
        if self.disable_cli_model_fetch and (has_paths or has_dir):
            raise ValueError(
                "Apps with disable_cli_model_fetch=True must not set "
                "model_file_paths or model_file_dir."
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
    supported_devices: list[Device] = Field(default_factory=list)
    app_repo_relative_path: str | None = (
        None  # relative path within qualcomm/ai-hub-apps
    )

    @model_validator(mode="after")
    def _validate_repo(self) -> "QAIHAAppInfo":
        if not self.app_repo_url and not self.app_repo_relative_path:
            raise ValueError(
                f"App '{self.id}': one of app_repo_url or app_repo_relative_path must be set"
            )
        return self

    # License information
    license_url: str
    license_type: AppLicense

    ##########################
    # Build System Information
    ##########################

    base_docker: str | None = None  # Dockerfile filename relative to tools/docker/

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
