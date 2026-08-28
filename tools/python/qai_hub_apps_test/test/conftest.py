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
    config.addinivalue_line(
        "markers",
        "verify_related_models: validate each app's related_models against AI Hub "
        "Models metadata (run with -m verify_related_models)",
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
    parser.addoption(
        "--no-docker",
        action="store_true",
        default=False,
        help="Build natively on the host instead of inside a Docker container",
    )
    parser.addoption(
        "--cli-version",
        default=None,
        help="qai-hub-apps CLI version to install on the QDC device (default: latest)",
    )
    parser.addoption(
        "--cli-source",
        default="source",
        choices=["source", "s3", "prod"],
        help="Where to install the CLI from on the QDC device: source (a wheel built "
        "from this checkout, default), s3 (nightly index), or prod (PyPI)",
    )
    parser.addoption(
        "--cli-wheel",
        default=None,
        help="Prebuilt CLI wheel to install on the device; if omitted, one is built "
        "(source) or downloaded (s3/prod) per --cli-source",
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


@pytest.fixture(scope="session")
def use_docker(request: pytest.FixtureRequest) -> bool:
    return not request.config.getoption("--no-docker")


@pytest.fixture(scope="session")
def cli_version(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--cli-version")


@pytest.fixture(scope="session")
def cli_source(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--cli-source")


@pytest.fixture(scope="session")
def cli_bundle(
    request: pytest.FixtureRequest,
    cli_source: str,
    cli_version: str | None,
    test_stage: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, str] | None:
    """The (wheel, registry) the device installs; resolved once per session.

    The wheel is built (source) or downloaded (s3/prod); the registry is resolved to
    match it. Returns None when the on-device stage is skipped.
    """
    if test_stage in ("fetch", "build"):
        return None
    from qai_hub_apps_test.qdc.app_test_job import obtain_cli_bundle

    out_dir = tmp_path_factory.mktemp("cli_wheel")
    override = request.config.getoption("--cli-wheel")
    return obtain_cli_bundle(cli_source, cli_version, str(out_dir), wheel=override)
