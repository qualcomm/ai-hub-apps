# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from pathlib import Path

from packaging.version import parse as parse_version

from qai_hub_apps._version import __version__
from qai_hub_apps.logging_utils import set_log_level

set_log_level()

# registry.yaml is excluded from the wheel, so its presence means a source checkout.
_BUNDLED_REGISTRY = Path(__file__).parent / "registry.yaml"
PACKAGE_NAME = "qai-hub-apps"


def _is_dev(version: str | None = None) -> bool:
    """Return True if this is a development (pre-release) build.

    Parameters
    ----------
    version:
        Version to check. If None, checks the installed version, and additionally
        treats a source checkout (bundled registry.yaml present) as a dev build --
        a checkout sitting on a release tag has a non-dev version.

    Returns
    -------
    bool
        True if this is a development build.
    """
    if version is not None:
        return parse_version(version).is_devrelease
    return parse_version(__version__).is_devrelease or _BUNDLED_REGISTRY.exists()


__all__ = ["PACKAGE_NAME", "__version__", "_is_dev"]
