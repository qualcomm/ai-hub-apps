# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.model_metadata import ModelMetadata


def load_class_labels(labels_path: str | Path) -> list[str]:
    """Load class display names from a labels file.

    Parameters
    ----------
    labels_path
        Path to the labels file, one class name per line, ordered by class
        index.

    Returns
    -------
    list[str]
        Class display names ordered by class index.
    """
    with open(labels_path) as labels_file:
        return [line.rstrip("\n") for line in labels_file if line.strip()]


def load_labels(models_dir: Path, metadata: ModelMetadata) -> list[str]:
    """Load class labels for the bundled model.

    Labels ship with the model asset as ``labels.txt``.
    """
    if metadata.labels_file:
        labels_path = models_dir / metadata.labels_file
        if labels_path.exists():
            return load_class_labels(labels_path)
    # TODO: v0.65.0 doesn't have labels bundled with the model; revert when ai-hub-models-cli is bumped
    return load_class_labels(models_dir / "kinetics400_labels.txt")


def top_k_predictions(
    scores: np.ndarray, labels: list[str], k: int
) -> list[tuple[str, float]]:
    """Return the ``k`` highest-scoring classes with their scores.

    Parameters
    ----------
    scores
        1-D array of per-class scores, length ``len(labels)``.
    labels
        Class display names ordered by class index.
    k
        Number of top predictions to return.

    Returns
    -------
    list[tuple[str, float]]
        The top-``k`` (class name, score) pairs, highest score first.
    """
    top_indices = np.argsort(scores)[::-1][:k]
    return [(labels[i], float(scores[i])) for i in top_indices]
