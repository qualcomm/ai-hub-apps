# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Access the target device that ``qai-hub-apps run`` configured for this run."""

from __future__ import annotations

import os
from dataclasses import dataclass

_ENV_NAME = "QAI_HUB_APPS_DEVICE_NAME"
_ENV_HTP_VERSION = "QAI_HUB_APPS_HEXAGON_VERSION"
_ENV_OS = "QAI_HUB_APPS_DEVICE_OS"
_ENV_CHIPSET = "QAI_HUB_APPS_CHIPSET"
_ENV_SOC_MODEL = "QAI_HUB_APPS_SOC_MODEL"


@dataclass(frozen=True)
class Device:
    name: str
    htp_version: str
    os: str
    chipset: str
    soc_model: int


def get_current_device() -> Device | None:
    """Return the device the CLI configured for this run, or None if unset.

    The values come from the ``QAI_HUB_APPS_*`` environment variables that
    ``qai-hub-apps run`` injects; None means the app was not launched through it.
    """
    name = os.environ.get(_ENV_NAME)
    if name is None:
        return None
    soc_model = os.environ.get(_ENV_SOC_MODEL)
    return Device(
        name=name,
        htp_version=os.environ.get(_ENV_HTP_VERSION, ""),
        os=os.environ.get(_ENV_OS, ""),
        chipset=os.environ.get(_ENV_CHIPSET, ""),
        soc_model=int(soc_model) if soc_model else 0,
    )
