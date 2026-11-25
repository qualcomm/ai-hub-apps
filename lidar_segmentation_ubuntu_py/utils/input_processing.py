# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import utils.constants as C

_MEANS = np.array(C.INPUT_MEANS, dtype=np.float32)
_STDS = np.array(C.INPUT_STDS, dtype=np.float32)


def load_scan(scan_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a KITTI-format Velodyne scan.

    The file is a flat float32 array of ``(x, y, z, remission)`` records.

    Parameters
    ----------
    scan_path
        Path to the ``.bin`` scan.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Points of shape [num_points, 3] and remissions of shape [num_points],
        both float32.

    Raises
    ------
    ValueError
        If the file length is not a whole number of records.
    """
    raw = np.fromfile(scan_path, dtype=np.float32)
    if raw.size == 0 or raw.size % C.SCAN_FIELDS != 0:
        raise ValueError(
            f"{scan_path} is not a KITTI Velodyne scan: {raw.size} float32 values "
            f"is not a multiple of {C.SCAN_FIELDS} (x, y, z, remission)."
        )
    scan = raw.reshape(-1, C.SCAN_FIELDS)
    return scan[:, :3], scan[:, 3]


@dataclass(frozen=True)
class RangeProjection:
    """A LiDAR scan projected onto the model's spherical range image.

    Attributes
    ----------
    model_input
        The normalized projection, shaped and laid out for the model.
    mask
        True where a pixel received a point, shape [height, width].
    column
        Per-point range-image column, shape [num_points], int32.
    row
        Per-point range-image row, shape [num_points], int32.
    """

    model_input: np.ndarray
    mask: np.ndarray
    column: np.ndarray
    row: np.ndarray


def project_scan(
    points: np.ndarray,
    remissions: np.ndarray,
    height: int,
    width: int,
    channels_first: bool,
) -> RangeProjection:
    """Project a point cloud into the model's input tensor.

    Each point's azimuth picks its column and its elevation its row, so the
    scan becomes a dense 2-D image the convolutional model can consume. Points
    are written farthest-first, leaving the nearest return in every pixel, and
    pixels with no return stay zero.

    Parameters
    ----------
    points
        Point coordinates of shape [num_points, 3], float32.
    remissions
        Per-point remission (intensity) of shape [num_points], float32.
    height
        Rows of the range image, one per laser beam.
    width
        Columns of the range image, spanning 360 degrees of azimuth.
    channels_first
        True if the model expects NCHW, False for NHWC.

    Returns
    -------
    RangeProjection
        The model input tensor, the occupancy mask, and each point's pixel.
    """
    fov_down = C.FOV_DOWN_DEG / 180.0 * np.pi
    fov = abs(fov_down) + abs(C.FOV_UP_DEG / 180.0 * np.pi)

    depth = np.sqrt(np.einsum("ij,ij->i", points, points))
    yaw = -np.arctan2(points[:, 1], points[:, 0])
    pitch = np.arcsin(points[:, 2] / depth)

    column = np.clip(np.floor(0.5 * (yaw / np.pi + 1.0) * width), 0, width - 1).astype(
        np.int32
    )
    row = np.clip(
        np.floor((1.0 - (pitch + abs(fov_down)) / fov) * height), 0, height - 1
    ).astype(np.int32)

    channels = np.empty((len(points), C.NUM_INPUT_CHANNELS), dtype=np.float32)
    channels[:, 0] = depth
    channels[:, 1:4] = points
    channels[:, 4] = remissions
    channels -= _MEANS
    channels /= _STDS

    order = np.argsort(depth)[::-1]
    rows, columns = row[order], column[order]

    image = np.zeros((height, width, C.NUM_INPUT_CHANNELS), dtype=np.float32)
    image[rows, columns] = channels[order]

    # Upstream is (proj_idx > 0) -- laserscan.py:194 at the pinned commit -- so
    # the pixel won by point 0 is zeroed. Kept for parity with the weights.
    index = np.full((height, width), -1, dtype=np.int32)
    index[rows, columns] = order
    mask = index > 0
    image *= mask[:, :, None]

    model_input = (
        np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        if channels_first
        else image[None]
    )
    return RangeProjection(model_input=model_input, mask=mask, column=column, row=row)
