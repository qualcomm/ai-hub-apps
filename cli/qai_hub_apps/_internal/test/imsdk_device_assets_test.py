# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json

import pytest

from qai_hub_apps._internal.utils.paths import SHARED_UTILS_ROOT
from qai_hub_apps._internal.utils.versions import load_versions
from qai_hub_apps.utils.devices import device_env, resolve_device_info

pytestmark = pytest.mark.bundler_unit

_ASSETS = json.loads(
    (
        SHARED_UTILS_ROOT / "scripts" / load_versions()["IMSDK_DEVICE_ASSETS_FILE"]
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("device_name", sorted(_ASSETS))
def test_imsdk_device_assets(device_name: str) -> None:
    env = device_env(resolve_device_info(device_name))
    assert env["QAI_HUB_APPS_DEVICE_NAME"] == device_name
    assert set(_ASSETS[device_name]) == {env["QAI_HUB_APPS_DEVICE_OS"]}

    for assets in _ASSETS[device_name].values():
        assert set(assets) == {"env", "cdi"}
        for asset in assets.values():
            assert set(asset) == {"url", "path", "sha256"}
