# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
import urllib.error
from pathlib import Path

from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.errors import QAIHubAppsError
from qai_hub_apps.registry import Registry

logger = logging.getLogger(__name__)


def run_fetch(
    app_id: str,
    output_dir: Path,
    registry: Registry,
    model_asset: ModelAsset | None = None,
) -> None:
    try:
        app_dest = registry.fetch_app(app_id, output_dir, model_asset=model_asset)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        msg = (
            f"Download failed (HTTP {e.code})"
            if isinstance(e, urllib.error.HTTPError)
            else f"Download failed: {e.reason}"
        )
        raise QAIHubAppsError(msg) from e
    logger.info("Extracted to %s", app_dest.as_posix())
    print(app_dest)
