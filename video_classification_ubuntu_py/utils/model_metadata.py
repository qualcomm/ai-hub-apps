# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelMetadata:
    """Model I/O contract read from the ``metadata.json`` shipped with the asset.

    Attributes
    ----------
    model_filename
        Name of the model file within the models directory.
    input_shape
        Input tensor shape, ``[1, T, H, W, C]`` or ``[1, C, T, H, W]`` for a
        video (clip) model.
    labels_file
        Name of the labels file shipped alongside the model (relative to the
        models directory), if the model's output spec declares one. ``None``
        if the model doesn't carry its own labels via metadata.
    """

    model_filename: str
    input_shape: list[int]
    labels_file: str | None

    @property
    def is_channels_first(self) -> bool:
        """True when the channel axis precedes the frame axis."""
        return self.input_shape[1] == 3

    @property
    def num_frames(self) -> int:
        """Frames the model consumes per inference."""
        return self.input_shape[2] if self.is_channels_first else self.input_shape[1]

    @property
    def crop_size(self) -> tuple[int, int]:
        """Spatial (height, width) the model expects."""
        if self.is_channels_first:
            return self.input_shape[-2], self.input_shape[-1]
        return self.input_shape[-3], self.input_shape[-2]


def load_model_metadata(models_dir: Path) -> ModelMetadata:
    """Load the model I/O contract from ``<models_dir>/metadata.json``.

    Only the first (and only) model file, its first input, and first output
    are used, matching a single-input/single-output classifier.

    Parameters
    ----------
    models_dir
        Directory containing the model file and its ``metadata.json``.

    Returns
    -------
    ModelMetadata
        The model filename, input shape, and labels file (if any).

    Raises
    ------
    FileNotFoundError
        If ``metadata.json`` is not present in ``models_dir``.
    """
    metadata_path = models_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_path}. It ships alongside the "
            "model in the AI Hub asset bundle; place it in the models directory "
            "next to the model file."
        )

    with metadata_path.open() as f:
        metadata = json.load(f)

    model_filename, model_spec = next(iter(metadata["model_files"].items()))
    input_spec = next(iter(model_spec["inputs"].values()))
    output_spec = next(iter(model_spec["outputs"].values()))

    return ModelMetadata(
        model_filename=model_filename,
        input_shape=input_spec["shape"],
        labels_file=output_spec.get("labels_file"),
    )
