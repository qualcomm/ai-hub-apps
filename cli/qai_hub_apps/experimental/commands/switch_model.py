# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.errors import InvalidArgumentError
from qai_hub_apps.experimental.commands.build import _resolve_app_from_dir
from qai_hub_apps.registry import Registry

logger = logging.getLogger(__name__)


def run_switch_model(
    app_path: Path, registry: Registry, model_asset: ModelAsset
) -> bool:
    """Replace the model bundled in an already-fetched app; return whether it changed."""
    app_dir = app_path.resolve()
    if not app_dir.is_dir():
        raise InvalidArgumentError(
            f"'{app_path}' is not a directory. Pass a directory produced by "
            "'qai-hub-apps fetch'."
        )
    app = _resolve_app_from_dir(app_dir, registry)
    if not app.switch_model(app_dir, model_asset):
        logger.info("'%s' already bundles %s; nothing to switch.", app.id, model_asset)
        return False
    logger.info(
        "Rebuilding '%s' may be required for the new model to take effect: "
        "qai-hub-apps build %s --clean",
        app.id,
        app_dir.as_posix(),
    )
    return True
