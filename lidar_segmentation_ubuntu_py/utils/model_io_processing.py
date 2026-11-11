# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

import utils.constants as C
from utils.input_processing import RangeProjection


def decode_class_map(model_output: np.ndarray) -> np.ndarray:
    """Convert the model output into a per-pixel class-id range image.

    The exported model emits per-class scores; the predicted class is the
    per-pixel argmax over the class axis. That axis can only be the last (NHWC)
    or the first (NCHW) axis of the batch-stripped tensor, since a middle axis
    is always spatial.

    Parameters
    ----------
    model_output
        Per-class scores of shape [1, num_classes, height, width] (NCHW) or
        [1, height, width, num_classes] (NHWC).

    Returns
    -------
    np.ndarray
        Class-id range image of shape (height, width), dtype uint8.

    Raises
    ------
    ValueError
        If neither the first nor the last axis holds the class scores.
    """
    output = model_output[0]

    if output.shape[-1] == C.NUM_CLASSES:
        class_axis = output.ndim - 1
    elif output.shape[0] == C.NUM_CLASSES:
        class_axis = 0
    else:
        raise ValueError(
            f"Unexpected model output shape {model_output.shape}: neither the "
            f"first nor the last axis equals {C.NUM_CLASSES} classes."
        )

    return np.argmax(output, axis=class_axis).astype(np.uint8)


def unproject_labels(class_map: np.ndarray, projection: RangeProjection) -> np.ndarray:
    """Read each point's predicted class back out of the range image.

    Parameters
    ----------
    class_map
        Class-id range image of shape (height, width), dtype uint8.
    projection
        The projection the class map was predicted from, holding each point's
        pixel.

    Returns
    -------
    np.ndarray
        Per-point class ids of shape [num_points], dtype uint8.
    """
    return class_map[projection.row, projection.column]
