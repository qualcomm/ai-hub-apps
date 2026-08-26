# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Verify each app's ``related_models`` list against AI Hub Models metadata.

For each app it:

1. FAILS if any listed ``related_models`` entry does not ship an asset matching
   the app's ``runtime`` + ``precisions`` + ``supported_devices``. Export-only
   models (no downloadable asset but exportable from source) always pass.
2. REPORTS — without failing — models that share the seed's use case and match
   the same filters but are not yet listed.

The seed is ``related_models[0]`` (the model built and run on-device by CI).
Apps with ``skip_related_models_verify`` set are skipped.
Usage: ``pytest -m verify_related_models``.
"""

from __future__ import annotations

import pytest
from packaging.version import Version
from qai_hub_models_cli.proto.info_pb2 import ModelUseCase
from qai_hub_models_cli.proto_helpers.manifest import get_manifest, get_manifest_entry
from qai_hub_models_cli.proto_helpers.platform import get_platform
from qai_hub_models_cli.proto_helpers.release_assets import (
    AssetNotFoundError,
    filter_release_assets,
    get_model_release_assets,
)
from qai_hub_models_cli.versions import CURRENT_VERSION

from qai_hub_apps_test.configs.info_yaml import AppStatus, QAIHAAppInfo
from qai_hub_apps_test.utils.paths import get_all_apps

pytestmark = pytest.mark.verify_related_models


def _matches_app(
    model_id: str,
    version: Version,
    runtime: list[str] | None,
    precision: list[str] | None,
    devices: list[str] | None,
    skip_export_only: bool,
) -> bool:
    """Whether a model ships an asset matching the app's runtime/precision/devices.

    Export-only models (no downloadable asset but exportable from source) are skipped if *skip_export_only* = True
    """
    try:
        assets = get_model_release_assets(model_id, version=version)
    except AssetNotFoundError as e:
        return not skip_export_only and e.model_sharing_restricted
    except KeyError:
        return False
    try:
        filtered = filter_release_assets(
            assets,
            get_platform(version=version),
            runtime=runtime,
            precision=precision,
            device=devices,
        )
    except KeyError:
        return False
    return bool(filtered.assets)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize ``app_info`` with every app that declares the fixture."""
    if "app_info" not in metafunc.fixturenames:
        return
    apps = [QAIHAAppInfo.from_app(p)[0] for p in get_all_apps()]
    metafunc.parametrize("app_info", apps, ids=[a.id for a in apps])


def test_related_models(app_info: QAIHAAppInfo) -> None:
    if app_info.skip_related_models_verify:
        pytest.skip(app_info.skip_related_models_verify)
    if app_info.status == AppStatus.PUBLISHED_WEBSITE_ONLY:
        pytest.skip(
            "app is published_website_only; apps are not part of cli for automated verification"
        )
    if not app_info.related_models:
        pytest.fail(f"{app_info.id}: related_models is empty")

    version = (
        CURRENT_VERSION
        if app_info.qaihm_version is None
        else Version(app_info.qaihm_version)
    )

    runtime = [str(r) for r in app_info.runtime] or None
    precision = [str(p) for p in app_info.precisions] or None
    devices = [str(d.name) for d in app_info.supported_devices] or None
    listed = [str(m) for m in app_info.related_models]

    # Every listed model must match the app's runtime/precision/devices.
    # Export-only models are kept as-is (they remain valid app choices).
    invalid = [
        m
        for m in listed
        if not _matches_app(
            m, version, runtime, precision, devices, skip_export_only=False
        )
    ]

    # Report same-use-case models that match but are not listed.
    seed = listed[0]  # use related_models[0] as seed as that is tested on-device
    listed_set = set(listed)
    try:
        seed_use_case = get_manifest_entry(seed, version=version).use_case
    except KeyError:
        seed_use_case = ModelUseCase.MODEL_USE_CASE_UNSPECIFIED

    suggestions: list[str] = []
    # use_case was added to the manifest in a later release; older versions leave
    # it UNSPECIFIED for every model, so skip suggestions when it isn't set.
    if seed_use_case != ModelUseCase.MODEL_USE_CASE_UNSPECIFIED:
        for entry in get_manifest(version).models:
            if entry.id in listed_set or entry.use_case != seed_use_case:
                continue
            # Skip export-only models when suggesting
            if _matches_app(
                entry.id, version, runtime, precision, devices, skip_export_only=True
            ):
                suggestions.append(entry.id)

    if suggestions:
        print(
            f"\n{app_info.id}: {len(suggestions)} unlisted model(s) match "
            f"runtime={runtime} precision={precision} devices={devices} "
            f"and share the seed's use case:\n  {', '.join(sorted(suggestions))}\n"
            "Consider adding them to related_models if appropriate."
        )

    assert not invalid, (
        f"{app_info.id}: invalid related_models entries with no AI Hub Models asset "
        f"matching the app's runtime={runtime}, precision={precision}, "
        f"supported_devices={devices}: {invalid}.\n"
        f"To fix, in apps/{app_info.id}/info.yaml remove each invalid entry, or "
        "replace it with a model that both matches this app's "
        "runtime/precision/device and has been verified to run with the app."
    )
