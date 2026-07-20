# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest

from qai_hub_apps.configs.model_asset import ModelAsset


def test_chipset_and_device_mutually_exclusive():
    with pytest.raises(ValueError, match="at most one of 'chipset' or 'device'"):
        ModelAsset(model_id="m", chipset="chip-1", device="Device A")
