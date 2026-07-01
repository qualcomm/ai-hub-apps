# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from packaging.version import parse as parse_version

from qai_hub_apps._version import __version__
from qai_hub_apps.logging_utils import set_log_level

set_log_level()


def _is_dev(version: str = __version__) -> bool:
    """Return True if ``version`` is a development (pre-release) build.

    Defaults to the installed package version.
    """
    return parse_version(version).is_devrelease


__all__ = ["__version__", "_is_dev"]
