# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "device_test: on-device app tests using the CLI and QDC (run with -m device_test)",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model-selection",
        default=None,
        choices=["first", "random", "all"],
        help="Which model(s) to test per app: first, random, or all (required for device_test)",
    )
    parser.addoption(
        "--qdc-token",
        default=None,
        help="QDC API token for on-device test submission",
    )
    parser.addoption(
        "--test-stage",
        default="all",
        choices=["fetch", "build", "all"],
        help="Furthest stage to run: fetch, build, or all (includes on-device QDC submission)",
    )


@pytest.fixture(scope="session")
def model_selection(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--model-selection")


@pytest.fixture(scope="session")
def qdc_token(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--qdc-token")


@pytest.fixture(scope="session")
def test_stage(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--test-stage")
