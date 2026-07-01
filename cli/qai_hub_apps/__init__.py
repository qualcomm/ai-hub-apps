# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

from packaging.version import parse as parse_version

from qai_hub_apps._version import __version__
from qai_hub_apps.logging_utils import set_log_level

# TODO: remove this once 0.57.0 lands on PyPI.
os.environ.setdefault("QAIHM_CLI_FORCE_VERSION", "0.56.0")

set_log_level()


def _is_dev(version: str = __version__) -> bool:
    """Return True if ``version`` is a development (pre-release) build.

    Defaults to the installed package version.
    """
    return parse_version(version).is_devrelease


__all__ = ["__version__", "_is_dev"]
