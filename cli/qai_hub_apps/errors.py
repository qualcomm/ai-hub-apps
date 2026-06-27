# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path


class QAIHubAppsError(Exception):
    """Base exception for all qai-hub-apps CLI errors."""


class AppNotFoundError(QAIHubAppsError):
    def __init__(self, app_id: str) -> None:
        self.app_id = app_id
        super().__init__(
            f"App '{app_id}' not found. Run 'qai-hub-apps list' to see available apps."
        )


class AppIncompatibleError(QAIHubAppsError):
    pass


class InvalidArgumentError(QAIHubAppsError):
    pass


class RegistryNotFoundError(QAIHubAppsError):
    def __init__(self, path: Path | str) -> None:
        super().__init__(
            f"Registry not found: {Path(path).as_posix()}\nTip: pass --registry PATH"
        )


class RegistryFetchError(QAIHubAppsError):
    def __init__(self, url: str) -> None:
        super().__init__(
            f"Failed to fetch registry from {url}\n"
            "Check your internet connection or try reinstalling: pip install -U qai-hub-apps"
        )


class ModelAssetNotFoundError(QAIHubAppsError):
    def __init__(
        self,
        model_id: str,
        chipset: str | None = None,
        *,
        reason: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.chipset = chipset
        detail = f" for chipset '{chipset}'" if chipset else ""
        message = f"Model asset '{model_id}'{detail} not found."
        message += (
            f" {reason}"
            if reason
            else " Check that the model ID and chipset are correct."
        )
        super().__init__(message)
