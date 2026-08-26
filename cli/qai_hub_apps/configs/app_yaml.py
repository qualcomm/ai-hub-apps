# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from enum import Enum, unique

from pydantic import field_validator

from qai_hub_apps.configs.base_config import BaseConfig


@unique
class AppStatus(Enum):
    UNPUBLISHED = "unpublished"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


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


class EnvironmentConfig(BaseConfig):
    python_version: str | None = None
    requirements_file: str = "requirements.txt"
    apt: list[str] = []


class AppUrl(BaseConfig):
    source: str


class AppInfo(BaseConfig):
    name: str
    id: str
    status: AppStatus
    headline: str
    description: str
    domain: str
    use_case: str
    app_repo_url: str
    app_type: AppType
    runtime: list[str]
    related_models: list[str]
    precisions: list[str]
    languages: list[AppLanguage] = []
    supported_devices: list[str] = []
    model_file_paths: list[str] = []
    model_file_dir: str | None = None
    disable_cli_model_fetch: bool = False
    environment: EnvironmentConfig | None = None
    url: AppUrl | None = None
    deprecation_notice: str | None = None
    qaihm_version: str | None = None

    @field_validator("runtime", mode="before")
    @classmethod
    def _normalize_runtime(cls, value: object) -> object:
        """Normalize runtime; a single value is wrapped into a list."""
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("supported_devices", mode="before")
    @classmethod
    def _normalize_supported_devices(cls, value: object) -> object:
        """Normalize devices to names; mapping entries keep only their ``name``.

        Registries may list a device either as a plain name or as a mapping with
        a ``name`` and internal test ``status``.
        """
        if isinstance(value, list):
            return [v.get("name") if isinstance(v, dict) else v for v in value]
        return value
