# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
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


def run_switch(
    app_path: Path,
    registry: Registry,
    model_asset: ModelAsset | None,
) -> None:
    """Swap the model bundled in an already-fetched app directory."""
    logger.debug("run_switch: app_path=%s, model_asset=%s", app_path, model_asset)
    if not app_path.is_dir():
        raise InvalidArgumentError(
            f"'{app_path}' is not a directory. Pass a directory produced by "
            "'qai-hub-apps fetch'."
        )
    app_dir = app_path.resolve()
    app = _resolve_app_from_dir(app_dir, registry)

    if model_asset is None:
        raise InvalidArgumentError(
            f"No model provided to switch '{app.id}' to. Pass --model / --model-id "
            "/ --model-path.\n"
            f"See the models this app supports:\n  qai-hub-apps info {app.id}"
        )

    model_id = app.switch_model(app_dir, model_asset)
    logger.info("Switched '%s' to model '%s'.", app.id, model_id)
    logger.info(
        "Any existing build output for '%s' may now be stale; run build/run with "
        "--clean to rebuild with the new model.",
        app.id,
    )
