# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest

import qai_hub_apps.registry.remote as remote_mod
from qai_hub_apps.errors import RegistryFetchError
from qai_hub_apps.registry.remote import (
    _get_cached_registry_path,
    _registry_s3_url,
    ensure_registry,
)


@pytest.fixture(autouse=True)
def clear_ensure_registry_cache():
    ensure_registry.cache_clear()
    yield
    ensure_registry.cache_clear()


def test_registry_s3_url_contains_version():
    url = _registry_s3_url("1.2.3")
    assert "1.2.3" in url
    assert url.endswith("/registry.yaml")
    assert url.startswith("https://")
    assert "/releases/1.2.3/" in url


def test_registry_s3_url_uses_dev_prefix_for_dev_build():
    url = _registry_s3_url("0.32.0.dev27+gabc1234")
    assert quote("/releases/dev/0.32.0.dev27+gabc1234/") in url


def test_get_cached_registry_path_includes_version(tmp_path):
    with patch(
        "qai_hub_apps.registry.remote.user_cache_dir", return_value=str(tmp_path)
    ):
        path = _get_cached_registry_path("1.2.3")
    assert "1.2.3" in str(path)
    assert path.name == "registry.yaml"


def test_ensure_registry_returns_bundled_when_present(tmp_path):
    bundled = tmp_path / "registry.yaml"
    bundled.write_text("bundled: true")
    with (
        patch.object(remote_mod, "_BUNDLED_REGISTRY", bundled),
        patch(
            "qai_hub_apps.registry.remote.urllib.request.urlretrieve"
        ) as mock_retrieve,
    ):
        result = ensure_registry("1.2.3")
    mock_retrieve.assert_not_called()
    assert result == bundled


def test_ensure_registry_uses_cache_when_present(tmp_path):
    no_bundled = tmp_path / "nonexistent.yaml"
    #  non-existent path, don't pick up the real bundled file from the dev checkout
    with (
        patch.object(remote_mod, "_BUNDLED_REGISTRY", no_bundled),
        patch(
            "qai_hub_apps.registry.remote.user_cache_dir", return_value=str(tmp_path)
        ),
    ):
        cache_path = _get_cached_registry_path("1.2.3")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("cached: true")

        with patch(
            "qai_hub_apps.registry.remote.urllib.request.urlretrieve"
        ) as mock_retrieve:
            result = ensure_registry("1.2.3")

        mock_retrieve.assert_not_called()
        assert result == cache_path


def test_ensure_registry_downloads_when_not_cached(tmp_path):
    no_bundled = tmp_path / "nonexistent.yaml"
    with (
        patch.object(remote_mod, "_BUNDLED_REGISTRY", no_bundled),
        patch(
            "qai_hub_apps.registry.remote.user_cache_dir", return_value=str(tmp_path)
        ),
    ):
        cache_path = _get_cached_registry_path("1.2.3")

        def fake_retrieve(url: str, dest: str) -> None:
            Path(dest).write_text("downloaded: true")

        with patch(
            "qai_hub_apps.registry.remote.urllib.request.urlretrieve",
            side_effect=fake_retrieve,
        ):
            result = ensure_registry("1.2.3")

        assert result == cache_path
        assert cache_path.exists()


def test_ensure_registry_raises_on_network_failure(tmp_path):
    no_bundled = tmp_path / "nonexistent.yaml"
    with (
        patch.object(remote_mod, "_BUNDLED_REGISTRY", no_bundled),
        patch(
            "qai_hub_apps.registry.remote.user_cache_dir", return_value=str(tmp_path)
        ),
        patch(
            "qai_hub_apps.registry.remote.urllib.request.urlretrieve",
            side_effect=OSError("network error"),
        ),
        pytest.raises(RegistryFetchError),
    ):
        ensure_registry("1.2.3")
