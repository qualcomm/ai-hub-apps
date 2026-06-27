# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pydantic import ConfigDict

from qai_hub_apps_test.configs.base_config import BaseConfig
from qai_hub_apps_test.configs.info_yaml import QAIHACLIAppInfo
from qai_hub_apps_test.utils.paths import REPOSITORY_ROOT

_DEFAULT_REGISTRY: AppRegistry | None = None


class AppRegistry(BaseConfig):
    """
    Registry of all published AI Hub Apps.
    Produced at release time and bundled with the CLI.
    """

    model_config = ConfigDict(extra="ignore", revalidate_instances="always")

    schema_version: str
    min_cli_version: str
    version: str | None = None
    apps: list[QAIHACLIAppInfo]

    @staticmethod
    def load() -> AppRegistry:
        """
        Load the default app registry.
        The object is a singleton and will only be created from disk once.
        """
        global _DEFAULT_REGISTRY  # noqa: PLW0603
        if not _DEFAULT_REGISTRY:
            _DEFAULT_REGISTRY = AppRegistry.from_yaml(REPOSITORY_ROOT / "registry.yaml")
        return _DEFAULT_REGISTRY
