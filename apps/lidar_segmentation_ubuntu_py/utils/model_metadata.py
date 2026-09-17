# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import utils.constants as C


@dataclass(frozen=True)
class ModelMetadata:
    """Model I/O contract read from the ``metadata.json`` shipped with the asset.

    Attributes
    ----------
    model_filename
        Name of the model file within the models directory (e.g. ``salsanext.tflite``).
    input_shape
        Input tensor shape, [batch, height, width, channels] (NHWC) or
        [batch, channels, height, width] (NCHW).
    output_shape
        Output tensor shape, [batch, height, width, num_classes] (NHWC) or
        [batch, num_classes, height, width] (NCHW).
    """

    model_filename: str
    input_shape: list[int]
    output_shape: list[int]

    def __post_init__(self) -> None:
        if (
            len(self.input_shape) != 4
            or C.NUM_INPUT_CHANNELS not in self.input_shape[1:]
        ):
            raise ValueError(
                f"Expected a 4-D input with a {C.NUM_INPUT_CHANNELS}-channel axis, "
                f"got {self.input_shape}."
            )

    @property
    def channels_first(self) -> bool:
        """True if the input is NCHW rather than NHWC."""
        return self.input_shape.index(C.NUM_INPUT_CHANNELS, 1) == 1

    @property
    def input_height(self) -> int:
        """Rows of the spherical projection the model consumes."""
        return self.input_shape[2] if self.channels_first else self.input_shape[1]

    @property
    def input_width(self) -> int:
        """Columns of the spherical projection the model consumes."""
        return self.input_shape[3] if self.channels_first else self.input_shape[2]


def load_model_metadata(models_dir: Path) -> ModelMetadata:
    """Load the model I/O contract from ``<models_dir>/metadata.json``.

    ``metadata.json`` ships alongside every AI Hub model asset. Only the first
    model file, its first input and first output are used, matching a
    single-input/single-output segmentation model.

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
