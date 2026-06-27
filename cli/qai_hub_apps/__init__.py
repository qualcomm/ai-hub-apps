# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

from packaging.version import parse as parse_version

from qai_hub_apps._version import __version__

# TODO: remove this once 0.57.0 lands on PyPI.
os.environ.setdefault("QAIHM_CLI_FORCE_VERSION", "0.56.0")


def _is_dev() -> bool:
    """Return True if the current install is a development (pre-release) build."""
    return parse_version(__version__).is_devrelease


__all__ = ["__version__", "_is_dev"]
