# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Experimental (preview) CLI features.

Everything under this package is opt-in and may change or be removed without
notice.
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argparse import _SubParsersAction
else:
    _SubParsersAction = Any

#: Environment variable that enables experimental features.
ENV_VAR = "QAI_HUB_APPS_EXPERIMENTAL"

_TAG = "[experimental]"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def add_experimental_parser(
    subparsers: _SubParsersAction,
    name: str,
    **kwargs: Any,
) -> argparse.ArgumentParser:
    """Register an experimental subcommand.

    The subcommand is registered only when experimental features are enabled.
    Otherwise, a dummy ``argparse.ArgumentParser`` is returned so callers may
    continue configuring the parser unconditionally.
    """
    if not is_enabled():
        return argparse.ArgumentParser(add_help=False)  # pragma: no cover

    if help_text := kwargs.get("help"):
        kwargs["help"] = f"{help_text} {_TAG}"

    return subparsers.add_parser(name, **kwargs)


def is_enabled() -> bool:
    """Return whether experimental features are enabled.

    Reads the ``QAI_HUB_APPS_EXPERIMENTAL`` environment variable.
    """
    return os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY
