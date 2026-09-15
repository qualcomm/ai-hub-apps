# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelIO:
    """One model file's I/O contract, read from the asset's ``metadata.json``.

    Attributes
    ----------
    filename
        Name of the model file within the models directory.
    input_shape
        Input tensor shape, 4-D with a channel dim of 3 (NHWC or NCHW).
    output_names
        Output tensor names, in the order the model declares them.
    """

    filename: str
    input_shape: list[int]
    output_names: list[str]

    def __post_init__(self) -> None:
        if len(self.input_shape) != 4 or 3 not in self.input_shape[1:]:
            raise ValueError(
                "Expected a 4-D input shape with a channel dim of 3 (NHWC or "
                f"NCHW); got {self.input_shape} for {self.filename}"
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


def load_model_metadata(models_dir: Path) -> list[ModelIO]:
    """Load the I/O contract of every model file in ``<models_dir>/metadata.json``.

    ``metadata.json`` ships alongside every AI Hub model asset.

    Parameters
    ----------
    models_dir
        Directory containing the model files and their ``metadata.json``.

    Returns
    -------
    list[ModelIO]
        One entry per model file, in the order ``metadata.json`` declares them.

    Raises
    ------
    FileNotFoundError
        If ``metadata.json`` is not present in ``models_dir``.
    """
    metadata_path = models_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_path}. It ships alongside the "
            "models in the AI Hub asset bundle; place it in the models directory "
            "next to the model files."
        )

    with metadata_path.open() as f:
        metadata = json.load(f)

    return [
        ModelIO(
            filename=filename,
            input_shape=next(iter(spec["inputs"].values()))["shape"],
            output_names=list(spec["outputs"]),
        )
        for filename, spec in metadata["model_files"].items()
    ]


def find_model(models: Sequence[ModelIO], output_names: Sequence[str]) -> ModelIO:
    """Pick the model whose outputs are exactly *output_names*.

    Parameters
    ----------
    models
        Candidate models, as returned by :func:`load_model_metadata`.
    output_names
        Output tensor names identifying the wanted model.

    Returns
    -------
    ModelIO
        The matching model.

    Raises
    ------
    ValueError
        If no model has exactly these outputs.
    """
    wanted = set(output_names)
    for model in models:
        if set(model.output_names) == wanted:
            return model
    available = {m.filename: m.output_names for m in models}
    raise ValueError(
        f"No model in the bundle has outputs {sorted(wanted)}. Available: {available}"
    )
