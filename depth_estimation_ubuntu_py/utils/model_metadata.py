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
    model_id
        AI Hub model id, e.g. ``depth_anything_v3``.
    model_filename
        Name of the model file within the models directory (e.g.
        ``model.tflite``).
    input_shape
        Shape of the model's first input tensor.
    output_shape
        Shape of the model's first output tensor.
    """

    model_id: str
    model_filename: str
    input_shape: list[int]
    output_shape: list[int]


def load_model_metadata(models_dir: Path) -> ModelMetadata:
    """Load the model I/O contract from ``<models_dir>/metadata.json``.

    ``metadata.json`` ships alongside every AI Hub model asset.

    Only the first model file, its first input and its first output are read.

    Parameters
    ----------
    models_dir
        Directory containing the model file and its ``metadata.json``.

    Returns
    -------
    ModelMetadata
        The model id, filename and its input/output shapes.

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
        model_id=metadata.get("model_id", Path(model_filename).stem),
        model_filename=model_filename,
        input_shape=input_spec["shape"],
        output_shape=output_spec["shape"],
    )
