# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging

from qai_hub_models_cli.proto_helpers.platform import DeviceInfo
from qai_hub_models_cli.proto_helpers.platform_enums import os_proto_to_str
from qai_hub_models_cli.utils import build_table
from qai_hub_models_cli.versions import CURRENT_VERSION as QAIHM_VERSION

from qai_hub_apps import __version__
from qai_hub_apps.errors import InvalidArgumentError, QAIHubAppsError
from qai_hub_apps.user_config import (
    get_configured_device,
    set_configured_device,
)
from qai_hub_apps.utils.devices import (
    device_env,
    list_supported_devices,
    resolve_device_info,
)
from qai_hub_apps.utils.github import make_issue_url
from qai_hub_apps.validate.platform_check import get_host_info

logger = logging.getLogger(__name__)


def _prompt_for_device() -> DeviceInfo:
    """Prompt the user to pick a device from a numbered list; return it."""
    devices = list_supported_devices()
    if not devices:
        issue_url = make_issue_url(
            title="No supported devices are available to configure",
            body=(
                f"Version: {__version__}\n"
                f"AI Hub Models version: {QAIHM_VERSION}\n"
                f"{get_host_info()}"
            ),
        )
        raise QAIHubAppsError(
            "No supported devices are available to configure. This is likely a bug - please file an "
            f"issue and we'll look into it:\n  {issue_url}"
        )

    rows = [
        [str(i), d.name, os_proto_to_str(d.os), d.chipset]
        for i, d in enumerate(devices, start=1)
    ]
    print(
        build_table(
            ["#", "Name", "OS", "Chipset"],
            rows,
            wrap_column="Name",
            title="Select your device",
        )
    )

    try:
        choice = input(f"Enter a number [1-{len(devices)}]: ").strip()
    except (EOFError, KeyboardInterrupt) as e:
        raise InvalidArgumentError("No device selected.") from e

    if not choice.isdigit() or not (1 <= int(choice) <= len(devices)):
        raise InvalidArgumentError(
            f"Invalid selection '{choice}'; expected a number between 1 and "
            f"{len(devices)}."
        )
    return devices[int(choice) - 1]


def run_configure(device: str | None, show: bool = False) -> None:
    """Configure and persist the target device for ``run``.

    Parameters
    ----------
    device
        A device name to set non-interactively. If None (and not ``show``),
        prompt the user to pick one.
    show
        If True, print the currently configured device and return.
    """
    if show:
        current = get_configured_device()
        if current is None:
            print("No target device configured. Run 'qai-hub-apps configure'.")
        else:
            print(f"Configured device: {current.name}")
        return

    info = resolve_device_info(device) if device is not None else _prompt_for_device()
    path = set_configured_device(info)

    logger.info("Configured target device '%s' (%s).", info.name, path)
    logger.debug("'run' will inject the device environment: %s", device_env(info))
