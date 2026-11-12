# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from enum import Enum, unique
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from qai_hub_apps.configs.base_config import BaseConfig
from qai_hub_apps.configs.field_types import Device, Precision, TargetRuntime

# Per-app metadata file.
INFO_FILENAME = "info.yaml"


@unique
class AppStatus(Enum):
    UNPUBLISHED = "unpublished"  # WIP: not on the website or CLI, but tested in CI
    PUBLISHED = "published"  # Live on the website, in the CLI, and tested
    PUBLISHED_WEBSITE_ONLY = (
        "published_website_only"  # On the website only; not CLI-fetchable or tested
    )
    DEPRECATED = "deprecated"  # Like published, but carries a deprecation notice


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
    IMSDK = "imsdk"


class AppUrl(BaseConfig):
    source: str


class AppInfo(BaseConfig):
    """CLI-facing app info — the fields written to registry.yaml."""

    name: str
    id: str
    status: AppStatus
    headline: str
    description: str
    domain: str
    use_case: str
    app_repo_url: str
    app_type: AppType
    runtime: list[TargetRuntime]
    related_models: list[str]
    precisions: list[Precision]
    languages: list[AppLanguage] = []
    supported_devices: list[Device] = Field(default_factory=list)
    model_file_paths: list[Path] = []  # Destination paths for downloaded model files
    model_file_dir: str | None = None  # Directory to unzip all model files into
    disable_cli_model_fetch: bool = False  # Apps that fetch their model at runtime
    url: AppUrl | None = None
    # Shown by the CLI for DEPRECATED apps; a default is used when unset.
    deprecation_notice: str | None = None
    # Supported AI Hub Models version, used to download model assets.
    # If None, uses installed version as supported.
    qaihm_version: str | None = None

    @field_validator("runtime", mode="before")
    @classmethod
    def _normalize_runtime(cls, value: object) -> object:
        """Normalize runtime; a single value is wrapped into a list."""
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def _validate_model_location(self) -> AppInfo:
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
