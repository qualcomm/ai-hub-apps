# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np

import utils.constants as C
from utils.geometry import calc_location, calc_theta_ray


def load_labels(labels_path: str) -> list[str]:
    """Read the class label list shipped with the model asset.

    Parameters
    ----------
    labels_path
        Path to the labels file, one label per line.

    Returns
    -------
    list[str]
        Labels indexed by the detector's class index.
    """
    with open(labels_path) as f:
        return [line.strip() for line in f if line.strip()]


def _to_batched_input(rgb_image: np.ndarray, channels_first: bool) -> np.ndarray:
    """Normalize an RGB image to [0, 1] and add a batch dimension."""
    normalized = rgb_image.astype(np.float32) / 255.0
    if channels_first:
        normalized = normalized.transpose(2, 0, 1)
    return np.expand_dims(normalized, axis=0)


def preprocess_frame(
    rgb_frame: np.ndarray, height: int, width: int, channels_first: bool
) -> np.ndarray:
    """Resize a frame to the network resolution and normalize it to [0, 1].

    The resize does not preserve aspect ratio, matching the reference
    implementation.

    Parameters
    ----------
    rgb_frame
        RGB image of shape [H, W, 3] and dtype uint8.
    height
        Network input height.
    width
        Network input width.
    channels_first
        Whether the network expects NCHW rather than NHWC.

    Returns
    -------
    np.ndarray
        Batched float32 input in [0, 1].
    """
    resized = cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return _to_batched_input(resized, channels_first)


def select_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
    net_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> list[tuple[list[list[int]], float, int]]:
    """Rescale detector boxes to frame pixels and keep the top-scoring ones.

    Parameters
    ----------
    boxes
        Boxes of shape (N, 4) as (x1, y1, x2, y2) in network pixels.
    scores
        Scores of shape (N,).
    class_indices
        Class index of shape (N,).
    net_size
        Network input size as (height, width).
    frame_size
        Frame size as (height, width).

    Returns
    -------
    list[tuple[list[list[int]], float, int]]
        Up to ``MAX_DETECTIONS`` entries of (box as ``[[xmin, ymin], [xmax,
        ymax]]``, score, class index), highest score first. Boxes with a
        non-positive area are dropped.
    """
    net_height, net_width = net_size
    frame_height, frame_width = frame_size
    scaled = boxes.astype(np.float64)
    scaled[:, (0, 2)] *= frame_width / net_width
    scaled[:, (1, 3)] *= frame_height / net_height

    detections = []
    for index in np.argsort(-scores):
        x1, y1, x2, y2 = (int(v) for v in scaled[index])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame_width - 1), min(y2, frame_height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            ([[x1, y1], [x2, y2]], float(scores[index]), int(class_indices[index]))
        )
        if len(detections) == C.MAX_DETECTIONS:
            break
    return detections


def preprocess_crop(
    rgb_frame: np.ndarray,
    box_2d: list[list[int]],
    height: int,
    width: int,
    channels_first: bool,
) -> np.ndarray:
    """Crop a detected object and prepare it as 3D-head input.

    The resize does not preserve aspect ratio, matching the reference
    implementation.

    Parameters
    ----------
    rgb_frame
        RGB image of shape [H, W, 3] and dtype uint8.
    box_2d
        Box as ``[[xmin, ymin], [xmax, ymax]]`` in frame pixels.
    height
        Network input height.
    width
        Network input width.
    channels_first
        Whether the network expects NCHW rather than NHWC.

    Returns
    -------
    np.ndarray
        Batched float32 input in [0, 1].
    """
    (x1, y1), (x2, y2) = box_2d
    crop = rgb_frame[y1 : y2 + 1, x1 : x2 + 1]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
    return _to_batched_input(resized, channels_first)


def decode_3d_box(
    orient: np.ndarray,
    conf: np.ndarray,
    dim: np.ndarray,
    kitti_class: str,
    box_2d: list[list[int]],
    proj_matrix: np.ndarray,
    frame_width: int,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Turn one 3D-head prediction into an oriented 3D box.

    Parameters
    ----------
    orient
        Orientation prediction of shape (bins, 2), each row (cos, sin).
    conf
        Per-bin confidence of shape (bins,).
    dim
        Dimension residual of shape (3,), added to the class average.
    kitti_class
        KITTI class name whose average dimensions the residual applies to.
    box_2d
        Box as ``[[xmin, ymin], [xmax, ymax]]`` in frame pixels.
    proj_matrix
        3x4 camera-to-image projection matrix.
    frame_width
        Frame width in pixels.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray] | None
        Global orientation in radians, dimensions (height, width, length) in
        metres, and box center of shape (3,) in camera coordinates. None if the
        box center could not be solved for.
    """
    dimension = np.asarray(dim, dtype=np.float64) + C.CLASS_AVERAGE_DIMS[kitti_class]

    best_bin = int(np.argmax(conf))
    cos, sin = orient[best_bin]
    # Bin centers, matching the reference bin generation for `bins` bins.
    interval = 2 * np.pi / C.NUM_ORIENTATION_BINS
    bin_center = best_bin * interval + interval / 2
    alpha = float(np.arctan2(sin, cos)) + bin_center - np.pi

    theta_ray = calc_theta_ray(frame_width, box_2d, proj_matrix)
    location = calc_location(dimension, proj_matrix, box_2d, alpha, theta_ray)
    if location is None:
        return None
    return alpha + theta_ray, dimension, location
