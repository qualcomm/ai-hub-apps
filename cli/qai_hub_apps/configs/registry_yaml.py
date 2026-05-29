# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import warnings

from packaging.version import Version
from pydantic import model_validator

from qai_hub_apps import __version__, _is_dev
from qai_hub_apps.configs.app_yaml import AppInfo
from qai_hub_apps.configs.base_config import BaseConfig

# this maps a SCHEMA version to the last supported CLI version
# key is schema version and value is a pip constraint for installation recommendation
SCHEMA_CLI_SUPPORT_MAP: dict[str, str] = {
    "1.0": "qai-hub-apps<0.30.0",
}

MIN_SUPPORTED_SCHEMA_VERSION = Version("1.1")


class AppRegistry(BaseConfig):
    schema_version: str
    min_cli_version: str
    version: str | None = None
    apps: list[AppInfo]

    @model_validator(mode="after")
    def _check_schema_version(self) -> AppRegistry:
        if Version(self.schema_version) < MIN_SUPPORTED_SCHEMA_VERSION:
            hint = SCHEMA_CLI_SUPPORT_MAP.get(self.schema_version)
            detail = (
                f"Please downgrade: pip install '{hint}'"
                if hint
                else "No compatible CLI version is known for this registry schema."
            )
            raise ValueError(
                f"Registry schema version {self.schema_version} is not supported by this CLI. "
                f"{detail}"
            )
        return self

    @model_validator(mode="after")
    def _unique_ids(self) -> AppRegistry:
        ids = [a.id for a in self.apps]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Registry contains duplicate app IDs: {sorted(dupes)}")
        return self

    @model_validator(mode="after")
    def _check_cli_version(self) -> AppRegistry:
        if self.version is None:
            msg = (
                "Registry has no version (dev registry). "
                "Use generate_registry --build_and_upload to produce a versioned registry."
            )
            if _is_dev():
                warnings.warn(msg, stacklevel=2)
            else:
                raise ValueError(
                    "Registry is missing a version. "
                    "The registry may be corrupt. Please upgrade: pip install -U qai-hub-apps"
                )
            return self

        cli = Version(__version__)

        # CLI is new enough to support the schema changes bundled in this registry.
        if cli <= Version(self.min_cli_version):
            msg = (
                f"This registry requires qai-hub-apps > {self.min_cli_version}, "
                f"but you have {__version__}. Please upgrade: pip install -U qai-hub-apps"
            )
            if _is_dev():
                warnings.warn(msg, stacklevel=2)
            else:
                raise ValueError(msg)

        # CLI is not loading a newer registry.
        if cli < Version(self.version):
            msg = (
                f"This registry was released with qai-hub-apps {self.version}, "
                f"but you have {__version__}. Please upgrade: pip install -U qai-hub-apps"
            )
            if _is_dev():
                warnings.warn(msg, stacklevel=2)
            else:
                raise ValueError(msg)

        return self
