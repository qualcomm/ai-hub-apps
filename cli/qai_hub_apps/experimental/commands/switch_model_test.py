# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.errors import InvalidArgumentError
from qai_hub_apps.experimental.commands.switch_model import run_switch_model
from qai_hub_apps.registry.base import App, Registry

MODEL = ModelAsset(model_id="m")


def test_switch_model_not_a_dir_raises(tmp_path, sample_registry_yaml):
    with pytest.raises(InvalidArgumentError, match="is not a directory"):
        run_switch_model(
            tmp_path / "missing", Registry.load(sample_registry_yaml), MODEL
        )


@pytest.mark.parametrize("switched", [True, False])
def test_switch_model_logs_outcome(
    switched, sample_app_dir, sample_registry_yaml, monkeypatch, caplog
):
    registry = Registry.load(sample_registry_yaml)
    app_dir = sample_app_dir(registry.find_by_id("test_app"))
    switch_model = MagicMock(return_value=switched)
    monkeypatch.setattr(App, "switch_model", switch_model)
    with caplog.at_level("INFO", logger="qai_hub_apps"):
        assert run_switch_model(app_dir, registry, MODEL) is switched
    switch_model.assert_called_once_with(app_dir.resolve(), MODEL)
    assert ("may be required" in caplog.text) is switched
    assert ("nothing to switch" in caplog.text) is not switched
