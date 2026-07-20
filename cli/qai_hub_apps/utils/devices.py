# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Helpers for resolving AI Hub device and chipset names."""

from __future__ import annotations

from qai_hub_models_cli.proto_helpers.platform import get_platform, resolve_chipset


def device_to_chipset(device: str) -> str:
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
    return resolve_chipset(get_platform(), device=device).name  # pragma: no cover
