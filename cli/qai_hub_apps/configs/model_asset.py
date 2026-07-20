# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelAsset:
    """Identifies a model to bundle into an app.

    Either ``model_id`` (downloaded from AI Hub) or ``path`` (a locally-exported
    model directory or ``.zip``) is set. For a local export, ``model_id`` is
    derived from the export's ``metadata.json`` and ``chipset``/``device`` do
    not apply.

    ``chipset`` and ``device`` are mutually exclusive ways to target an asset.
    """

    model_id: str | None = None
    chipset: str | None = None
    device: str | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.chipset is not None and self.device is not None:
            raise ValueError("Provide at most one of 'chipset' or 'device'.")
