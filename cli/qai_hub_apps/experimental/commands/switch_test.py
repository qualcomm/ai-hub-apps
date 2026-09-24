# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.errors import InvalidArgumentError
from qai_hub_apps.experimental.commands import switch as switch_mod
from qai_hub_apps.experimental.commands.switch import run_switch
from qai_hub_apps.registry.base import Registry


def test_run_switch_switches_model_and_warns_about_stale_build(
    sample_app_dir, sample_registry_yaml, monkeypatch, caplog
):
    registry = Registry.load(sample_registry_yaml)
    app = registry.find_by_id("test_app")
    app_dir = sample_app_dir(app)
    switch_model = MagicMock(return_value="test_model")
    monkeypatch.setattr(app, "switch_model", switch_model)
    monkeypatch.setattr(
        switch_mod, "_resolve_app_from_dir", MagicMock(return_value=app)
    )
    asset = ModelAsset(model_id="test_model")

    run_switch(app_dir, registry, asset)

    switch_model.assert_called_once_with(app_dir.resolve(), asset)
    assert "Switched 'test_app' to model 'test_model'." in caplog.text
    assert "may now be stale" in caplog.text


def test_run_switch_rejects_missing_directory(tmp_path):
    with pytest.raises(InvalidArgumentError, match="is not a directory"):
        run_switch(tmp_path / "nope", MagicMock(), ModelAsset(model_id="m"))


def test_run_switch_requires_a_model(sample_app_dir, sample_registry_yaml, monkeypatch):
    registry = Registry.load(sample_registry_yaml)
    app = registry.find_by_id("test_app")
    app_dir = sample_app_dir(app)
    switch_model = MagicMock()
    monkeypatch.setattr(app, "switch_model", switch_model)
    monkeypatch.setattr(
        switch_mod, "_resolve_app_from_dir", MagicMock(return_value=app)
    )

    with pytest.raises(InvalidArgumentError, match="No model provided"):
        run_switch(app_dir, registry, None)
    switch_model.assert_not_called()
