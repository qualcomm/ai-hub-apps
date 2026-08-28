# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Helpers for resolving AI Hub device and chipset names."""

from __future__ import annotations

from qai_hub_models_cli.proto_helpers.platform import (
    DeviceInfo,
    get_platform,
    resolve_chipset,
    resolve_device,
)
from qai_hub_models_cli.proto_helpers.platform_enums import os_proto_to_str

from qai_hub_apps.errors import InvalidArgumentError


def device_to_chipset(device: str) -> str:  # pragma: no cover
    """Return the canonical chipset id for an AI Hub *device* name.

    Parameters
    ----------
    device
        An AI Hub device name (e.g. ``"Snapdragon 8 Elite QRD"``).

    Returns
    -------
    str
        The canonical chipset id for *device*.

    Raises
    ------
    KeyError
        If *device* is not a known AI Hub device.
    """
    platform = get_platform()
    return resolve_chipset(
        chipsets=platform.chipsets, devices=platform.devices, device=device
    ).name


def list_supported_devices() -> list[DeviceInfo]:  # pragma: no cover
    """Return every supported AI Hub device.

    Returns
    -------
    list[DeviceInfo]
        All supported devices.
    """
    return list(get_platform().devices)


def resolve_device_info(device: str) -> DeviceInfo:  # pragma: no cover
    """Return the AI Hub device matching *device*.

    Parameters
    ----------
    device
        A device name (matched case-insensitively/leniently).

    Returns
    -------
    DeviceInfo
        The resolved device.

    Raises
    ------
    InvalidArgumentError
        If *device* is not a known AI Hub device.
    """
    platform = get_platform()
    try:
        return resolve_device(platform.devices, device)
    except Exception as e:
        raise InvalidArgumentError(f"'{device}' is not a known AI Hub device.") from e


def device_env(device: DeviceInfo) -> dict[str, str]:  # pragma: no cover
    """Return the runtime environment variables describing *device*.

    Parameters
    ----------
    device
        The device to describe.

    Returns
    -------
    dict[str, str]
        Environment variables (``QAI_HUB_APPS_*``) describing the device's
        name, Hexagon version, OS, etc.
    """
    platform = get_platform()
    chipset = resolve_chipset(
        chipsets=platform.chipsets, devices=platform.devices, device=device.name
    )
    return {
        "QAI_HUB_APPS_DEVICE_NAME": device.name,
        "QAI_HUB_APPS_HEXAGON_VERSION": f"v{chipset.htp_version}",
        "QAI_HUB_APPS_DEVICE_OS": os_proto_to_str(device.os),
        "QAI_HUB_APPS_CHIPSET": chipset.name,
        "QAI_HUB_APPS_SOC_MODEL": str(chipset.soc_model),
    }
