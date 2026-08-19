# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import urllib.request
from pathlib import Path
from urllib.parse import quote

from platformdirs import user_cache_dir

from qai_hub_apps import _BUNDLED_REGISTRY, _is_dev
from qai_hub_apps.errors import RegistryFetchError

_S3_BASE = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com"
_S3_PREFIX = "qai-hub-apps/releases"
_S3_PREFIX_DEV = f"{_S3_PREFIX}/dev"


def _registry_s3_url(version: str) -> str:
    prefix = _S3_PREFIX_DEV if _is_dev(version) else _S3_PREFIX
    return f"{_S3_BASE}/{prefix}/{quote(version)}/registry.yaml"


def _get_cached_registry_path(version: str) -> Path:
    return Path(user_cache_dir("qai-hub-apps")) / version / "registry.yaml"


@functools.cache
def ensure_registry(version: str) -> Path:
    """Return path to registry.yaml.

    Resolution order:
    1. Bundled file (dev/editable install)
    2. Local cache
    3. Download from S3 and cache
    """
    if _BUNDLED_REGISTRY.exists():
        return _BUNDLED_REGISTRY
    cache_path = _get_cached_registry_path(version)
    if cache_path.exists():
        return cache_path
    url = _registry_s3_url(version)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
    except (urllib.error.URLError, OSError) as e:
        raise RegistryFetchError(url) from e
    return cache_path
