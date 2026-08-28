# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from qai_hub_apps.configs.app_yaml import AppLanguage, AppType
from qai_hub_apps.conftest import make_app_info, make_device
from qai_hub_apps.errors import AppIncompatibleError, QAIHubAppsError
from qai_hub_apps.experimental.validate import platform_check
from qai_hub_apps.experimental.validate.platform_check import (
    _is_wsl,
    ensure_build_supported,
    ensure_device_supported,
    ensure_docker_available,
    ensure_run_supported,
)
from qai_hub_apps.registry.base import App


def test_docker_missing_binary(monkeypatch):
    monkeypatch.setattr(platform_check.shutil, "which", lambda _: None)
    with pytest.raises(QAIHubAppsError, match="not found on PATH"):
        ensure_docker_available()


def test_docker_daemon_unreachable(monkeypatch):
    monkeypatch.setattr(platform_check.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        platform_check.subprocess,
        "run",
        MagicMock(side_effect=subprocess.CalledProcessError(1, "docker")),
    )
    with pytest.raises(QAIHubAppsError, match="daemon is not reachable"):
        ensure_docker_available()


def test_docker_available(monkeypatch):
    monkeypatch.setattr(platform_check.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(platform_check.subprocess, "run", MagicMock())
    ensure_docker_available()  # no raise


def test_windows_app_on_non_windows_raises(monkeypatch):
    monkeypatch.setattr(platform_check.sys, "platform", "linux")
    app = App(make_app_info(app_type=AppType.WINDOWS, languages=[AppLanguage.CPP]))
    with pytest.raises(AppIncompatibleError, match="can only be built on Windows"):
        ensure_build_supported(app, use_docker=True)


def test_windows_cpp_docker_on_arm64_raises(monkeypatch):
    monkeypatch.setattr(platform_check.sys, "platform", "win32")
    monkeypatch.setattr(platform_check.platform, "machine", lambda: "ARM64")
    app = App(make_app_info(app_type=AppType.WINDOWS, languages=[AppLanguage.CPP]))
    with pytest.raises(AppIncompatibleError, match="Windows container image"):
        ensure_build_supported(app, use_docker=True)


def test_android_on_windows_raises(monkeypatch):
    monkeypatch.setattr(platform_check.sys, "platform", "win32")
    app = App(make_app_info(app_type=AppType.ANDROID, languages=[AppLanguage.JAVA]))
    with pytest.raises(
        AppIncompatibleError, match="can only be built on Linux or under WSL"
    ):
        ensure_build_supported(app, use_docker=True)


def test_windows_cpp_docker_on_x86_checks_docker(monkeypatch):
    monkeypatch.setattr(platform_check.sys, "platform", "win32")
    monkeypatch.setattr(platform_check.platform, "machine", lambda: "AMD64")
    docker_check = MagicMock()
    monkeypatch.setattr(platform_check, "ensure_docker_available", docker_check)
    app = App(make_app_info(app_type=AppType.WINDOWS, languages=[AppLanguage.CPP]))
    ensure_build_supported(app, use_docker=True)
    docker_check.assert_called_once()


def test_ubuntu_docker_checks_docker(monkeypatch):
    monkeypatch.setattr(platform_check.sys, "platform", "linux")
    docker_check = MagicMock()
    monkeypatch.setattr(platform_check, "ensure_docker_available", docker_check)
    ensure_build_supported(App(make_app_info(app_type=AppType.UBUNTU)), use_docker=True)
    docker_check.assert_called_once()


def test_no_docker_skips_docker_check(monkeypatch):
    monkeypatch.setattr(platform_check.sys, "platform", "linux")
    docker_check = MagicMock()
    monkeypatch.setattr(platform_check, "ensure_docker_available", docker_check)
    ensure_build_supported(
        App(make_app_info(app_type=AppType.UBUNTU)), use_docker=False
    )
    docker_check.assert_not_called()


def test_is_wsl(monkeypatch):
    monkeypatch.setattr(
        platform_check.platform,
        "uname",
        lambda: MagicMock(release="6.8.0-45-generic"),
    )
    assert _is_wsl() is False


def test_run_android_requires_adb(monkeypatch):
    monkeypatch.setattr(platform_check.shutil, "which", lambda _: None)
    app = App(make_app_info(app_type=AppType.ANDROID, languages=[AppLanguage.JAVA]))
    with pytest.raises(AppIncompatibleError, match="requires 'adb'"):
        ensure_run_supported(app, make_device(), use_docker=False)


def test_run_windows_app_on_non_windows_raises(monkeypatch):
    monkeypatch.setattr(platform_check.sys, "platform", "linux")
    app = App(make_app_info(app_type=AppType.WINDOWS, languages=[AppLanguage.CPP]))
    with pytest.raises(AppIncompatibleError, match="can only be run on Windows"):
        ensure_run_supported(app, make_device(), use_docker=False)


def test_run_ubuntu_app_on_wsl_raises(monkeypatch):
    monkeypatch.setattr(platform_check.sys, "platform", "linux")
    monkeypatch.setattr(platform_check, "_is_wsl", lambda: True)
    app = App(make_app_info(app_type=AppType.UBUNTU))
    with pytest.raises(AppIncompatibleError, match="native Linux"):
        ensure_run_supported(app, make_device(), use_docker=False)


def test_run_ubuntu_docker_checks_docker(monkeypatch):
    monkeypatch.setattr(platform_check.sys, "platform", "linux")
    monkeypatch.setattr(platform_check, "_is_wsl", lambda: False)
    docker_check = MagicMock()
    monkeypatch.setattr(platform_check, "ensure_docker_available", docker_check)
    ensure_run_supported(
        App(make_app_info(app_type=AppType.UBUNTU)), make_device(), use_docker=True
    )
    docker_check.assert_called_once()


def test_ensure_device_supported_rejects_other_device():
    app = App(make_app_info(supported_devices=["Device A"]))
    with pytest.raises(AppIncompatibleError, match="does not support the configured"):
        ensure_device_supported(app, "Device B")
