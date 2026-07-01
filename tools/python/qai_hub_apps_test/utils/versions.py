# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

from packaging.version import parse as parse_version

from qai_hub_apps_test.utils.paths import APPS_ROOT

_DEFAULT_VERSIONS_ENV = APPS_ROOT / "_shared" / "scripts" / "versions.env"


def is_dev(version: str) -> bool:
    """Return True if ``version`` is a development (pre-)release."""
    return parse_version(version).is_devrelease


def load_versions(path: Path = _DEFAULT_VERSIONS_ENV) -> dict[str, str]:
    """Parse KEY="VALUE" lines from a versions.env file into a dict.

    Parameters
    ----------
    path:
        Path to the versions.env file. Defaults to
        ``apps/_shared/scripts/versions.env`` in the repository.

    Returns
    -------
    dict[str, str]
        Mapping of variable names to their string values (quotes stripped).
    """
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            versions[key.strip()] = value.strip().strip('"')
    return versions
