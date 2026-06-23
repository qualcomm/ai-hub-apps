# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""On-device app tests driven by the CLI and QDC.

Three stages, run in order by name prefix (test_1_, test_2_, test_3_):
  1. Fetch  — downloads app + model via `qai-hub-apps fetch`
  2. Build  — builds the app from the fetched source
  3. Device — submits the built app to QDC for on-device execution

Usage:
  pytest -m device_test --model-selection first --test-stage fetch
  pytest -m device_test --model-selection first --test-stage build
  pytest -m device_test --model-selection first --test-stage all --qdc-token $QDC_API_TOKEN
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest

from qai_hub_apps_test.builders import build_app
from qai_hub_apps_test.configs.info_yaml import QAIHAAppInfo

pytestmark = pytest.mark.device_test


def _select_models(app_info: QAIHAAppInfo, mode: str) -> list[str]:
    # Apps with disable_cli_model_fetch download their model at runtime; the model is
    # never fetched, so yield a single param to avoid redundant fetch/build/device runs.
    if app_info.disable_cli_model_fetch:
        return ["no_model"]
    models = app_info.related_models
    if not models:
        return []
    if mode == "first":
        return [models[0]]
    if mode == "random":
        return [random.choice(models)]
    # "all"
    return list(models)


def _build_params(model_selection: str) -> list[tuple[QAIHAAppInfo, str]]:
    """Build (app_info, model_id) pairs from the CLI registry.

    Uses Registry.load() from CLI. For each app in the registry, loads the full QAIHAAppInfo from
    the local apps/<id>/info.yaml to access internal fields.
    """
    from qai_hub_apps.registry import (
        Registry,  # lazy import, CLI not required for other tests
    )

    params: list[tuple[QAIHAAppInfo, str]] = []
    for app in Registry.load().apps:
        app_info, _ = QAIHAAppInfo.from_app(app.id)
        params.extend(
            (app_info, model_id)
            for model_id in _select_models(app_info, model_selection)
        )
    return params


def _param_id(val: tuple[QAIHAAppInfo, str]) -> str:
    app_info, model_id = val
    return f"{app_info.id}-{model_id}"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize app_to_test using the --model-selection CLI option.

    Only fires for test functions that declare the `app_to_test` fixture.
    """
    if "app_to_test" not in metafunc.fixturenames:
        return
    model_selection = metafunc.config.getoption("--model-selection")
    if model_selection is None:
        markexpr = getattr(metafunc.config.option, "markexpr", "") or ""
        if "device_test" in markexpr or not markexpr:
            raise ValueError(
                "--model-selection is required when running device tests "
                "(choices: first, random, all)"
            )
        return
    params = _build_params(model_selection)
    metafunc.parametrize("app_to_test", params, ids=[_param_id(p) for p in params])


@pytest.fixture(scope="session")
def fetched_dirs() -> dict[tuple[str, str], Path]:
    """Maps (app_id, model_id) -> fetched app directory."""
    return {}


@pytest.fixture(scope="session")
def built_dirs() -> dict[tuple[str, str], Path]:
    """Maps (app_id, model_id) -> ready-to-test app directory."""
    return {}


def test_1_fetch_app(
    app_to_test: tuple[QAIHAAppInfo, str],
    tmp_path_factory: pytest.TempPathFactory,
    fetched_dirs: dict,
) -> None:
    """Fetch app + model via qai-hub-apps CLI."""
    app_info, model_id = app_to_test

    if app_info.skip_test:
        pytest.skip(app_info.skip_test)
    if not app_info.supported_devices:
        pytest.fail(f"No supported_devices defined in {app_info.id}/info.yaml")

    out_parent = tmp_path_factory.mktemp(f"{app_info.id}__{model_id}")
    fetch_cmd = [
        "qai-hub-apps",
        "fetch",
        app_info.id,
        "--output-dir",
        str(out_parent),
    ]
    # Apps with disable_cli_model_fetch download their model at runtime — fetching
    # with --model errors, so omit it.
    if not app_info.disable_cli_model_fetch:
        fetch_cmd += [
            "--model",
            model_id,
            "--chipset",
            app_info.supported_devices[0].chipset,
        ]
    subprocess.run(fetch_cmd, check=True)
    fetched_dirs[(app_info.id, model_id)] = out_parent / app_info.id


def test_2_build_app(
    app_to_test: tuple[QAIHAAppInfo, str],
    fetched_dirs: dict,
    built_dirs: dict,
    test_stage: str,
    use_docker: bool,
) -> None:
    """Build the app from the fetched source."""
    app_info, model_id = app_to_test

    if test_stage == "fetch":
        pytest.skip("--test-stage=fetch; skipping build stage")

    app_dir = fetched_dirs.get((app_info.id, model_id))
    if app_dir is None:
        pytest.skip("Fetch stage did not succeed or was skipped")

    assert app_dir is not None
    try:
        build_app(app_info, app_dir, use_docker=use_docker)
    except NotImplementedError as e:
        pytest.skip(str(e))

    built_dirs[(app_info.id, model_id)] = app_dir


def test_3_on_device_app(
    app_to_test: tuple[QAIHAAppInfo, str],
    built_dirs: dict,
    qdc_token: str | None,
    test_stage: str,
) -> None:
    """Submit app to QDC for on-device execution."""
    app_info, model_id = app_to_test

    if test_stage in ("fetch", "build"):
        pytest.skip(f"--test-stage={test_stage}; skipping on-device stage")

    app_dir = built_dirs.get((app_info.id, model_id))
    if app_dir is None:
        pytest.skip("Build stage did not succeed or was skipped")

    if not qdc_token:
        pytest.fail("--qdc-token is required for on-device tests")
    if not app_info.supported_devices:
        pytest.fail(f"No supported_devices defined in {app_info.id}/info.yaml")

    device = app_info.supported_devices[0].reference_device_name

    assert qdc_token is not None
    assert app_dir is not None

    from qai_hub_apps_test.qdc.app_test_job import (  # lazy import, QDC SDK not required for other tests
        submit_app_bundle_to_qdc_device,
    )

    success = submit_app_bundle_to_qdc_device(
        api_token=qdc_token,
        device=device,
        app_dir=app_dir,
        use_docker=True,
        job_name=f"{app_info.id}-{model_id}",
    )
    assert success, f"QDC job failed for {app_info.id} with model {model_id}"
