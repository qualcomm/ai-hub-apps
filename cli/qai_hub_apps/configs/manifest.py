# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from typing_extensions import Self

logger = logging.getLogger(__name__)

# Provenance file written into every fetched app dir.
MANIFEST_FILENAME = "qai_hub_apps.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelProvenance:
    """Records the model placed in an app and the files it put there.

    NOTE: ``files`` exists because the asset's ``metadata.json`` does not list
    every file packaged in the download (e.g. weight sidecars), so it cannot be
    used to clean up a previous model. Once metadata.json is complete this can
    be dropped in favour of deriving the file list from it:
    https://github.com/qcom-ai-hub/tetracode/issues/19248
    """

    model_id: str
    files: list[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=_now)


@dataclass
class Manifest:
    """Provenance for a fetched app: the versions that fetched it and its model."""

    cli_version: str | None = None
    qai_hub_models_version: str | None = None
    registry_version: str | None = None
    fetched_at: str | None = None
    model: ModelProvenance | None = None

    @classmethod
    def load(cls, app_dir: Path) -> Self:
        """Read the manifest from *app_dir*, or an empty one if it is absent or unreadable."""
        path = app_dir / MANIFEST_FILENAME
        try:
            data = json.loads(path.read_text())
            model = data.pop("model", None)
            manifest = cls(**data)
            if model is not None:
                manifest.model = ModelProvenance(**model)
        except (OSError, ValueError, TypeError):
            logger.debug("No usable manifest at %s", path)
            return cls()
        return manifest

    def write(self, app_dir: Path) -> None:
        """Write this manifest into *app_dir*."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        path = app_dir / MANIFEST_FILENAME
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        logger.debug("Wrote manifest %s: %s", MANIFEST_FILENAME, data)
