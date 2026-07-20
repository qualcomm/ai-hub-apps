# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Validated field types used by the app config models.

Concrete ``ProtoToken`` subclasses for platform fields such as precision and
runtime. Each holds the canonical lowercase token (e.g. ``"float"``,
``"onnx"``) and validates/normalizes its input on assignment, so invalid
values are rejected and equivalent spellings collapse to one form.

These are thin wrappers over the protobuf enums that ``qai_hub_models_cli``
provides, exposing them as plain string tokens usable as Pydantic fields.
"""

from __future__ import annotations

from typing import Any

from qai_hub_models_cli.proto_helpers.platform import get_platform, resolve_chipset
from qai_hub_models_cli.proto_helpers.platform_enums import (
    precision_proto_to_str,
    precision_str_to_proto,
    runtime_proto_to_str,
    runtime_str_to_proto,
)

from qai_hub_apps_test.configs.proto_token import ProtoToken


class Precision(ProtoToken):
    """Model precision token (e.g. ``"float"``, ``"w8a8"``)."""

    __slots__ = ()

    @staticmethod
    def _normalize(value: Any) -> str:
        return precision_proto_to_str(precision_str_to_proto(value))


class TargetRuntime(ProtoToken):
    """Target runtime token (e.g. ``"tflite"``, ``"onnx"``)."""

    __slots__ = ()

    @staticmethod
    def _normalize(value: Any) -> str:
        return runtime_proto_to_str(runtime_str_to_proto(value))


class Device(ProtoToken):
    """An AI Hub device name validated against the platform registry.

    The token is the device name itself (e.g. ``"Snapdragon X Elite CRD"``);
    ``chipset`` resolves its canonical chipset ID on demand. ``get_platform()``
    fetches and caches the platform registry, so repeated lookups don't
    re-fetch.
    """

    __slots__ = ()

    @staticmethod
    def _normalize(value: Any) -> str:
        # resolve_chipset raises KeyError for an unknown device; ProtoToken
        # surfaces that as a validation error.
        platform = get_platform()
        resolve_chipset(
            chipsets=platform.chipsets, devices=platform.devices, device=str(value)
        )
        return str(value)

    @property
    def reference_device_name(self) -> str:
        """The device name (kept for parity with prior device objects)."""
        return str(self)

    @property
    def chipset(self) -> str:
        """Canonical chipset ID for this device (e.g. ``"qualcomm-snapdragon-x-elite"``)."""
        platform = get_platform()
        return resolve_chipset(
            chipsets=platform.chipsets, devices=platform.devices, device=str(self)
        ).name
