# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from qai_hub_models_cli.proto.platform_pb2 import (
    DeviceInfo,
    OperatingSystem,
    OperatingSystemType,
)

from qai_hub_apps.configs.app_yaml import AppInfo, AppLanguage, AppType
from qai_hub_apps.registry.base import Registry


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: end-to-end tests that exercise the full CLI stack (run with -m integration)",
    )


def make_app_info(**overrides) -> AppInfo:
    """Factory for AppInfo with sensible defaults. Accepts keyword overrides."""
    defaults: dict = dict(
        name="Test App",
        id="test_app",
        status="published",
        headline="Test headline",
        description="Test description",
        domain="Test",
        use_case="Testing",
        app_repo_url="https://github.com/test/app",
        app_type=AppType.UBUNTU,
        runtime="onnx",
        related_models=["test_model"],
        precisions=["float"],
        languages=[AppLanguage.PYTHON],
        url=None,
    )
    defaults.update(overrides)
    return AppInfo(**defaults)


def make_device(
    name: str = "Snapdragon 8 Elite QRD",
    ostype: int = OperatingSystemType.OPERATING_SYSTEM_TYPE_ANDROID,
    os_version: str = "15",
    chipset: str = "qualcomm-snapdragon-8-elite",
) -> DeviceInfo:
    """Factory for a DeviceInfo proto with sensible defaults."""
    return DeviceInfo(
        name=name,
        chipset=chipset,
        os=OperatingSystem(ostype=ostype, version=os_version),
    )


@pytest.fixture(autouse=True)
def reset_registry_singleton():
    """Clear the Registry singleton between tests to prevent cross-test contamination."""
    Registry._instance = None
    yield
    Registry._instance = None


@pytest.fixture(autouse=True)
def configure_test_logging():
    """Run the package logger at debug level so caplog sees every record."""
    logger = logging.getLogger("qai_hub_apps")
    saved_level = logger.level
    logger.setLevel(logging.DEBUG)
    yield
    logger.setLevel(saved_level)


@pytest.fixture
def sample_app_info() -> AppInfo:
    return make_app_info()


@pytest.fixture
def sample_registry_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid registry.yaml to a temp file and return its path."""
    from qai_hub_apps import __version__

    content = f"""\
schema_version: '1.1'
min_cli_version: 0.0.1
version: '{__version__}'
apps:
- name: Test App
  id: test_app
  status: published
  headline: Test headline
  description: Test description
  domain: Test
  use_case: Testing
  app_repo_url: https://github.com/test/app
  app_type: ubuntu
  runtime: onnx
  related_models: [test_model]
  precisions: [float]
  languages: [Python]
"""
    p = tmp_path / "registry.yaml"
    p.write_text(content)
    return p
