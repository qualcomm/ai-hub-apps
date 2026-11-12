# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_apps.configs.app_yaml import AppType
from qai_hub_apps.conftest import make_app_info
from qai_hub_apps.registry.base import App
from qai_hub_apps.validate import is_app_supported
from qai_hub_apps.validate.platform_check import check_platform


def _make_app(app_type: AppType) -> App:
    return App(make_app_info(app_type=app_type))


def test_android_always_false_on_linux(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.validate.platform_check.sys.platform", "linux")
    assert check_platform(_make_app(AppType.ANDROID)) is False


def test_android_always_false_on_win32(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.validate.platform_check.sys.platform", "win32")
    assert check_platform(_make_app(AppType.ANDROID)) is False


def test_windows_on_win32(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.validate.platform_check.sys.platform", "win32")
    assert check_platform(_make_app(AppType.WINDOWS)) is True


def test_windows_on_linux(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.validate.platform_check.sys.platform", "linux")
    assert check_platform(_make_app(AppType.WINDOWS)) is False


def test_ubuntu_on_linux(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.validate.platform_check.sys.platform", "linux")
    assert check_platform(_make_app(AppType.UBUNTU)) is True


def test_ubuntu_on_win32(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.validate.platform_check.sys.platform", "win32")
    assert check_platform(_make_app(AppType.UBUNTU)) is False


def test_imsdk_on_linux(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.validate.platform_check.sys.platform", "linux")
    assert check_platform(_make_app(AppType.IMSDK)) is True


def test_imsdk_on_win32(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.validate.platform_check.sys.platform", "win32")
    assert check_platform(_make_app(AppType.IMSDK)) is False


def test_app_with_unknown_type_is_true():
    app = _make_app(AppType.UBUNTU)
    app.app_type = "unknown_platform"  # type: ignore[attr-defined]
    assert check_platform(app) is True


def test_is_app_supported_delegates_to_check_platform(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.validate.platform_check.sys.platform", "linux")
    app = _make_app(AppType.UBUNTU)
    assert is_app_supported(app) == check_platform(app)
