# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

from packaging.version import Version
from packaging.version import parse as parse_version
from platformdirs import user_cache_dir

from qai_hub_apps import __version__

logger = logging.getLogger(__name__)

_PACKAGE = "qai-hub-apps"
_PYPI_URL = f"https://pypi.org/pypi/{_PACKAGE}/json"
_CACHE_MAX_AGE_SECONDS = 3 * 24 * 60 * 60  # 3 days


def _cache_path() -> Path:  # pragma: no cover
    return Path(user_cache_dir(_PACKAGE)) / "latest-version.txt"


def _fetch_latest_version() -> Version:  # pragma: no cover
    """Fetch the latest (non-prerelease) published version from PyPI."""
    with urllib.request.urlopen(_PYPI_URL, timeout=10) as resp:
        data = json.load(resp)
    releases = data.get("releases", {})
    versions = [
        parsed for v in releases if not (parsed := parse_version(v)).is_prerelease
    ]
    return max(versions)


def _get_latest_version() -> Version:  # pragma: no cover
    """Return the latest published version, using a disk cache (~3 days)."""
    cache = _cache_path()
    if (
        cache.exists()
        and (time.time() - cache.stat().st_mtime) < _CACHE_MAX_AGE_SECONDS
    ):
        return parse_version(cache.read_text().strip())

    latest = _fetch_latest_version()
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(str(latest))
    except OSError:
        pass
    return latest


def check_for_update() -> None:
    """Log a notice if a newer ``qai-hub-apps`` release is available on PyPI.

    Never raises — any error is swallowed.
    """
    try:
        latest = _get_latest_version()
    except Exception:
        logger.debug("Update check failed", exc_info=True)
        return
    if latest > parse_version(__version__):
        logger.info(
            "A newer version of %s is available (%s); you have %s. "
            "\nUpgrade: pip install --upgrade %s",
            _PACKAGE,
            latest,
            __version__,
            _PACKAGE,
        )
