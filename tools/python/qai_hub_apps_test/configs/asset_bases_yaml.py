# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_apps_test.configs.base_config import BaseConfig
from qai_hub_apps_test.utils.paths import APPS_ROOT

_DEFAULT_ASSET_BASES: AssetBases | None = None


class AssetBases(BaseConfig):
    """
    Base URLs for AI Hub Apps assets.
    Loaded from apps/asset_bases.yaml.
    """

    app_store_url: str  # S3 base URL for app static assets
    app_repo_base: str  # GitHub repo base (no ref, no /apps suffix)

    @staticmethod
    def load() -> AssetBases:
        """
        Load the default asset bases config.
        The object is a singleton and will only be created from disk once.
        """
        global _DEFAULT_ASSET_BASES  # noqa: PLW0603
        if not _DEFAULT_ASSET_BASES:
            _DEFAULT_ASSET_BASES = AssetBases.from_yaml(APPS_ROOT / "asset_bases.yaml")
        return _DEFAULT_ASSET_BASES
