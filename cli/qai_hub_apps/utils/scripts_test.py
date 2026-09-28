# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qai_hub_apps.utils import scripts
from qai_hub_apps.utils.scripts import confirm, script_env, set_non_interactive


def test_set_non_interactive_auto_approves(monkeypatch):
    mock_input = MagicMock()
    monkeypatch.setattr("builtins.input", mock_input)
    set_non_interactive()

    assert confirm("Proceed?") is True
    mock_input.assert_not_called()
    assert script_env() == {scripts.NON_INTERACTIVE_ENV_VAR: "true"}


@pytest.mark.parametrize(("answer", "expected"), [("y", True), ("n", False)])
def test_confirm_reads_the_answer(monkeypatch, answer, expected):
    monkeypatch.setattr("builtins.input", lambda _: answer)
    assert confirm("Proceed?") is expected


def test_confirm_returns_default_with_no_answer(monkeypatch):
    """No tty (CI, piped input) falls back to the default."""
    monkeypatch.setattr("builtins.input", MagicMock(side_effect=EOFError))
    assert confirm("Proceed?") is False
