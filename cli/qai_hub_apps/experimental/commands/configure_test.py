# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qai_hub_apps.conftest import make_device
from qai_hub_apps.errors import InvalidArgumentError, QAIHubAppsError
from qai_hub_apps.experimental.commands import configure as configure_mod
from qai_hub_apps.experimental.commands.configure import (
    prompt_for_device,
    run_configure,
)

DEVICES = [make_device(name="Device A"), make_device(name="Device B")]


def test_prompt_for_device_no_devices_raises():
    with pytest.raises(QAIHubAppsError, match="No supported devices"):
        prompt_for_device([])


def test_prompt_for_device_returns_selection(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert prompt_for_device(DEVICES, title="Select your Android device") is DEVICES[0]
    out = capsys.readouterr().out
    assert "Select your Android device" in out
    assert "Device A" in out


def test_prompt_for_device_out_of_range_raises(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "3")
    with pytest.raises(InvalidArgumentError, match="Invalid selection"):
        prompt_for_device(DEVICES)


def test_prompt_for_device_aborted_raises(monkeypatch):
    monkeypatch.setattr("builtins.input", MagicMock(side_effect=KeyboardInterrupt))
    with pytest.raises(InvalidArgumentError, match="No device selected"):
        prompt_for_device(DEVICES)


def test_run_configure_show_configured(monkeypatch, capsys):
    monkeypatch.setattr(
        configure_mod, "get_configured_device", lambda: make_device(name="Device A")
    )
    run_configure(None, show=True)
    assert "Configured device: Device A" in capsys.readouterr().out


def test_run_configure_show_unconfigured(monkeypatch, capsys):
    monkeypatch.setattr(configure_mod, "get_configured_device", lambda: None)
    run_configure(None, show=True)
    assert "No target device configured" in capsys.readouterr().out


def test_run_configure_prompts_when_no_device(monkeypatch):
    monkeypatch.setattr(
        configure_mod, "list_supported_devices", MagicMock(return_value=DEVICES)
    )
    monkeypatch.setattr(
        configure_mod, "prompt_for_device", MagicMock(return_value=DEVICES[0])
    )
    monkeypatch.setattr(configure_mod, "device_env", MagicMock(return_value={}))
    set_device = MagicMock(return_value="/cfg/config.json")
    monkeypatch.setattr(configure_mod, "set_configured_device", set_device)
    run_configure(None)
    set_device.assert_called_once_with(DEVICES[0])
