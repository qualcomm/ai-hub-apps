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
        Name of the model file within the models directory (e.g. ``deeplab_xception.tflite``,
        ``fcn_resnet50.tflite``).
    input_shape
        Input tensor shape, NHWC: [batch, height, width, channels].
    output_shape
        Output tensor shape: [batch, height, width, num_classes] (NHWC) or
        [batch, num_classes, height, width] (NCHW). The class axis is resolved at
        decode time (see ``model_io_processing.decode_mask``).
    """

    model_filename: str
    input_shape: list[int]
    output_shape: list[int]

    @property
    def input_height(self) -> int:
        """Model input height (H in NHWC)."""
        return self.input_shape[1]

    @property
    def input_width(self) -> int:
        """Model input width (W in NHWC)."""
        return self.input_shape[2]


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
