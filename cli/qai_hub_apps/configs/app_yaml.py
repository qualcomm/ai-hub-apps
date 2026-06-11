# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from enum import Enum, unique

from qai_hub_models_cli.common import Precision, TargetRuntime

from qai_hub_apps.configs.base_config import BaseConfig


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


class EnvironmentConfig(BaseConfig):
    python_version: str | None = None
    requirements_file: str = "requirements.txt"
    apt: list[str] = []


class AppUrl(BaseConfig):
    source: str


class AppInfo(BaseConfig):
    name: str
    id: str
    status: str
    headline: str
    description: str
    domain: str
    use_case: str
    app_repo_url: str
    app_type: AppType
    runtime: TargetRuntime
    related_models: list[str]
    precisions: list[Precision]
    languages: list[AppLanguage] = []
    model_file_paths: list[str] = []
    model_file_dir: str | None = None
    disable_cli_model_fetch: bool = False
    environment: EnvironmentConfig | None = None
    url: AppUrl | None = None
