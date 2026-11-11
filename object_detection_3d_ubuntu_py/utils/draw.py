# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np

import utils.constants as C
from utils.geometry import create_corners, project_3d_points, rotation_matrix

# Edges of the 3D box in terms of create_corners' corner order.
_BOX_EDGES = (
    (0, 2),
    (4, 6),
    (0, 4),
    (2, 6),
    (1, 3),
    (1, 5),
    (7, 3),
    (7, 5),
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
)

# Diagonals across the front face, to show which way the object is facing.
_FRONT_EDGES = ((0, 3), (1, 2))


def draw_3d_box(
    rgb_frame: np.ndarray,
    proj_matrix: np.ndarray,
    orientation: float,
    dimension: np.ndarray,
    location: np.ndarray,
    label: str,
) -> None:
    """Draw a projected 3D bounding box onto a frame, in place.

    Parameters
    ----------
    rgb_frame
        RGB frame of shape [H, W, 3] and dtype uint8. Modified in place.
    proj_matrix
        3x4 camera-to-image projection matrix.
    orientation
        Global orientation of the object in radians.
    dimension
        Object dimensions as (height, width, length) in metres.
    location
        Box center of shape (3,) in camera coordinates.
    label
        Text drawn above the box.
    """
    corners = create_corners(dimension, location, rotation_matrix(orientation))
    points = project_3d_points(corners, proj_matrix)
    if points is None:
        return

    pixels = [(int(x), int(y)) for x, y in points]
    for start, end in _BOX_EDGES:
        cv2.line(rgb_frame, pixels[start], pixels[end], C.BOX_COLOR, C.BOX_THICKNESS)
    for start, end in _FRONT_EDGES:
        cv2.line(
            rgb_frame, pixels[start], pixels[end], C.BOX_FRONT_COLOR, C.BOX_THICKNESS
        )

    text_origin = (int(points[:, 0].min()), max(int(points[:, 1].min()) - 8, 12))
    cv2.putText(
        rgb_frame,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        C.LABEL_COLOR,
        1,
    )
