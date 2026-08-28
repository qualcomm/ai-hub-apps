# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import logging
from pathlib import Path

from platformdirs import user_config_dir
from qai_hub_models_cli.proto_helpers.platform import DeviceInfo

from qai_hub_apps import PACKAGE_NAME
from qai_hub_apps.errors import QAIHubAppsError
from qai_hub_apps.utils.devices import resolve_device_info

logger = logging.getLogger(__name__)

_DEVICE_KEY = "device"


def config_path() -> Path:
    """Return the path to the CLI's config file."""
    return Path(user_config_dir(PACKAGE_NAME)) / "config.json"


def _read_config() -> dict[str, object]:
    """Return the config dict, or an empty dict if missing/corrupt."""
    path = config_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.debug("No readable config at %s; treating as empty", path)
        return {}
    return data if isinstance(data, dict) else {}


def get_configured_device() -> DeviceInfo | None:
    """Return the configured target device, or None if not configured.

    Returns
    -------
    DeviceInfo | None
        The resolved target device, or None if none is configured.
    """
    name = _read_config().get(_DEVICE_KEY)
    logger.debug(f"Found configured device: {name}")
    if not isinstance(name, str):
        return None
    return resolve_device_info(name)


def set_configured_device(device: DeviceInfo) -> Path:
    """Set *device* as the target device, returning the config path.

    Parameters
    ----------
    device
        The device to persist as the target.

    Returns
    -------
    Path
        The path to the config file that was written.
    """
    path = config_path()
    config = _read_config()
    config[_DEVICE_KEY] = device.name
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except OSError as e:
        raise QAIHubAppsError(f"Could not write config to '{path}': {e}") from e
    logger.debug("Wrote device '%s' to %s", device.name, path)
    return path
