# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np


def load_class_labels(labels_path: str) -> list[str]:
    """Load AudioSet display names from the YamNet labels file.

    Parameters
    ----------
    labels_path
        Path to ``labels.txt``, one class display name per line, ordered by
        class index. Shipped alongside the model in the downloaded asset bundle.

    Returns
    -------
    list[str]
        Class display names ordered by class index.
    """
    with open(labels_path) as labels_file:
        return [line.rstrip("\n") for line in labels_file if line.strip()]


def top_k_labels(scores: np.ndarray, labels: list[str], k: int) -> list[str]:
    """Return the labels of the ``k`` highest-scoring classes.

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
    list[str]
        The top-``k`` class names, highest score first.
    """
    top_indices = np.argsort(scores)[::-1][:k]
    return [labels[i] for i in top_indices]
