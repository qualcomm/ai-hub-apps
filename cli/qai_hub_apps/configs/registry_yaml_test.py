# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from qai_hub_apps.configs.registry_yaml import AppRegistry
from qai_hub_apps.conftest import make_app_info


def _make_registry(**overrides) -> AppRegistry:
    defaults = dict(
        schema_version="1.1",
        min_cli_version="0.0.1",
        apps=[make_app_info()],
    )
    defaults.update(overrides)
    return AppRegistry(**defaults)


def test_old_schema_version_known_suggests_downgrade():
    with pytest.raises(ValidationError, match=r"qai-hub-apps<0.30.0"):
        _make_registry(schema_version="1.0")


def test_old_schema_version_unknown_gives_generic_message():
    with pytest.raises(ValidationError, match="No compatible CLI version is known"):
        _make_registry(schema_version="0.9")


def test_unique_ids_ok(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: True)
    app_a = make_app_info(id="app_a", name="App A")
    app_b = make_app_info(id="app_b", name="App B")
    registry = AppRegistry(
        schema_version="1.1", min_cli_version="0.0.1", apps=[app_a, app_b]
    )
    assert len(registry.apps) == 2


def test_unique_ids_raises_on_duplicate():
    app_a = make_app_info(id="same_id", name="App A")
    app_b = make_app_info(id="same_id", name="App B")
    with pytest.raises(ValidationError, match="same_id"):
        AppRegistry(schema_version="1.1", min_cli_version="0.0.1", apps=[app_a, app_b])


def test_no_version_dev_warns(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_registry(version=None)
    assert any("no version (dev registry)" in str(w.message).lower() for w in caught)


def test_no_version_prod_raises(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: False)
    with pytest.raises(ValidationError, match="Registry is missing a version"):
        _make_registry(version=None)


def test_version_ok_no_warning(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml.__version__", "1.0.0")
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_registry(version="1.0.0", min_cli_version="0.0.1")
    assert not any(issubclass(w.category, UserWarning) for w in caught)


def test_below_min_cli_version_dev_warns(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml.__version__", "0.5.0")
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_registry(version="0.5.0", min_cli_version="1.0.0")
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_below_min_cli_version_prod_raises(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml.__version__", "0.5.0")
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: False)
    with pytest.raises(ValidationError):
        _make_registry(version="0.5.0", min_cli_version="1.0.0")


def test_below_registry_version_dev_warns(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml.__version__", "0.5.0")
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_registry(version="1.0.0", min_cli_version="0.0.1")
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_below_registry_version_prod_raises(monkeypatch):
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml.__version__", "0.5.0")
    monkeypatch.setattr("qai_hub_apps.configs.registry_yaml._is_dev", lambda: False)
    with pytest.raises(ValidationError):
        _make_registry(version="1.0.0", min_cli_version="0.0.1")
