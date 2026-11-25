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
        Name of the model file within the models directory (e.g. ``sinet.tflite``).
    input_shape
        Input tensor shape, 4-D with a channel dim of 3 (NHWC or NCHW).
    output_shape
        Output tensor shape, 4-D.
    """

    model_filename: str
    input_shape: list[int]
    output_shape: list[int]

    def __post_init__(self) -> None:
        if len(self.input_shape) != 4 or 3 not in self.input_shape[1:]:
            raise ValueError(
                "Expected a 4-D input shape with a channel dim of 3 (NHWC or "
                f"NCHW); got {self.input_shape}"
            )

    @property
    def channels_first(self) -> bool:
        """Whether the input is laid out as NCHW rather than NHWC."""
        return self.input_shape.index(3, 1) == 1

    @property
    def input_height(self) -> int:
        """Model input height."""
        return self.input_shape[2] if self.channels_first else self.input_shape[1]

    @property
    def input_width(self) -> int:
        """Model input width."""
        return self.input_shape[3] if self.channels_first else self.input_shape[2]


def load_model_metadata(models_dir: Path) -> ModelMetadata:
    """Load the model I/O contract from ``<models_dir>/metadata.json``.

    ``metadata.json`` ships alongside every AI Hub model asset.

    Only the first (and only) model file, its first input, and first output are
    used, matching a single-input/single-output segmentation model.

    Parameters
    ----------
    models_dir
        Directory containing the model file and its ``metadata.json``.

    Returns
    -------
    ModelMetadata
        The model filename and its input/output shapes.

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
        output_shape=output_spec["shape"],
    )
