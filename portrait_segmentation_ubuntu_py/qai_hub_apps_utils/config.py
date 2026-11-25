# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""An app's own configuration, read from its ``info.yaml``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppInfo:
    """An app's ``info.yaml``.

    Attributes
    ----------
    name
        Display name, e.g. ``"Posenet Pose Estimation"``.
    use_case
        Task the app performs, e.g. ``"Pose Estimation"``.
    precisions
        Precisions the app ships, e.g. ``["w8a8"]``.
    runtime
        Inference runtime, e.g. ``"tflite"``.
    related_models
        AI Hub model ids the app wraps.
    model_file_paths
        Model paths relative to the app, as declared in ``info.yaml``. An app
        that picks its model at fetch time should prefer the filename it actually
        loaded over this list.
    """

    name: str = ""
    use_case: str = ""
    precisions: list[str] = field(default_factory=list)
    runtime: str = ""
    related_models: list[str] = field(default_factory=list)
    model_file_paths: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, app_root: Path | None = None) -> AppInfo:
        """Read ``info.yaml`` from `app_root`.

        Parameters
        ----------
        app_root
            The app's directory. Defaults to ``$QAIHA_APP_ROOT``, which the entry
            scripts export, else the current directory.

        Returns
        -------
        AppInfo
            The parsed info, or one with empty fields if the file is missing or invalid.
        """
        if app_root is None:
            app_root = Path(os.environ.get("QAIHA_APP_ROOT") or ".")
        try:
            parsed = yaml.safe_load(
                (app_root / "info.yaml").read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError):
            parsed = None
        if not isinstance(parsed, dict):
            return cls()
        return cls(
            name=str(parsed.get("name") or ""),
            use_case=str(parsed.get("use_case") or ""),
            precisions=[str(p) for p in parsed.get("precisions") or []],
            runtime=str(parsed.get("runtime") or ""),
            related_models=[str(m) for m in parsed.get("related_models") or []],
            model_file_paths=[str(p) for p in parsed.get("model_file_paths") or []],
        )
