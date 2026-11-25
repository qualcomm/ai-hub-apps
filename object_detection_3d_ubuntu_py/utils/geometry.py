# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Camera geometry for lifting a 2D box plus predicted orientation and
dimensions into a 3D box, following "3D Bounding Box Estimation Using Deep
Learning and Geometry" (https://arxiv.org/abs/1612.00496).
"""

from __future__ import annotations

import itertools

import numpy as np

# Corner sign pattern of create_corners: (x, y, z) each iterate over (+1, -1),
# x outermost.
_CORNER_SIGNS = np.array(list(itertools.product((1, -1), repeat=3)), dtype=np.float64)


def build_projection_matrix(
    frame_width: int, frame_height: int, hfov_deg: float
) -> np.ndarray:
    """Build a pinhole camera-to-image projection matrix.

    Parameters
    ----------
    frame_width
        Frame width in pixels.
    frame_height
        Frame height in pixels.
    hfov_deg
        Horizontal field of view in degrees.

    Returns
    -------
    np.ndarray
        3x4 projection matrix with the principal point at the frame center.
    """
    focal = (frame_width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array(
        [
            [focal, 0.0, frame_width / 2.0, 0.0],
            [0.0, focal, frame_height / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )


def calc_theta_ray(
    frame_width: int, box_2d: list[list[int]], proj_matrix: np.ndarray
) -> float:
    """Angle between the camera axis and the ray through the box center.

    Parameters
    ----------
    frame_width
        Frame width in pixels.
    box_2d
        Box as ``[[xmin, ymin], [xmax, ymax]]`` in frame pixels.
    proj_matrix
        3x4 camera-to-image projection matrix.

    Returns
    -------
    float
        Ray angle in radians, signed by which side of the frame center the box
        falls on.
    """
    fovx = 2 * np.arctan(frame_width / (2 * proj_matrix[0][0]))
    dx = (box_2d[1][0] + box_2d[0][0]) / 2 - frame_width / 2
    mult = -1.0 if dx < 0 else 1.0
    return float(np.arctan(2 * abs(dx) * np.tan(fovx / 2) / frame_width) * mult)


def rotation_matrix(yaw: float) -> np.ndarray:
    """Rotation about the vertical axis.

    Parameters
    ----------
    yaw
        Rotation in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def create_corners(
    dimension: np.ndarray, location: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Build the 8 corners of a 3D box.

    Parameters
    ----------
    dimension
        Object dimensions as (height, width, length) in metres.
    location
        Box center in camera coordinates.
    rotation
        3x3 rotation matrix applied before translation.

    Returns
    -------
    np.ndarray
        Corners of shape (8, 3) in camera coordinates.
    """
    half = np.array([dimension[2], dimension[0], dimension[1]]) / 2.0
    corners = _CORNER_SIGNS * half
    return corners @ rotation.T + np.asarray(location)


def project_3d_points(points: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray | None:
    """Project 3D camera-space points into image pixels.

    Parameters
    ----------
    points
        Points of shape (N, 3) in camera coordinates.
    proj_matrix
        3x4 camera-to-image projection matrix.

    Returns
    -------
    np.ndarray | None
        Integer pixel coordinates of shape (N, 2), or None if any point is at or
        behind the image plane, where the projection is not meaningful.
    """
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ proj_matrix.T
    depth = projected[:, 2]
    if np.any(depth <= 1e-6):
        return None
    return (projected[:, :2] / depth[:, None]).astype(np.int32)


def _corner_constraints(dimension: np.ndarray, alpha: float) -> list[np.ndarray]:
    """Enumerate the candidate (left, top, right, bottom) corner assignments.

    Which corner of the box touches each side of the 2D box depends on the
    object's relative angle; the reference implementation resolves this by trying
    every plausible combination and keeping the best fit.
    """
    dx, dy, dz = dimension[2] / 2, dimension[0] / 2, dimension[1] / 2

    left_mult, right_mult = 1.0, -1.0
    if np.deg2rad(88) < alpha < np.deg2rad(92):
        left_mult, right_mult = 1.0, 1.0
    elif np.deg2rad(-92) < alpha < np.deg2rad(-88):
        left_mult, right_mult = -1.0, -1.0
    elif -np.deg2rad(90) < alpha < np.deg2rad(90):
        left_mult, right_mult = -1.0, 1.0
    switch_mult = 1.0 if alpha > 0 else -1.0

    left = [[left_mult * dx, i * dy, -switch_mult * dz] for i in (-1, 1)]
    right = [[right_mult * dx, i * dy, switch_mult * dz] for i in (-1, 1)]
    top = [[i * dx, -dy, j * dz] for i in (-1, 1) for j in (-1, 1)]
    bottom = [[i * dx, dy, j * dz] for i in (-1, 1) for j in (-1, 1)]

    return [
        np.array(combo)
        for combo in itertools.product(left, top, right, bottom)
        if len({tuple(corner) for corner in combo}) == len(combo)
    ]


def calc_location(
    dimension: np.ndarray,
    proj_matrix: np.ndarray,
    box_2d: list[list[int]],
    alpha: float,
    theta_ray: float,
) -> np.ndarray | None:
    """Solve for the 3D box center that best reprojects onto the 2D box.

    Parameters
    ----------
    dimension
        Object dimensions as (height, width, length) in metres.
    proj_matrix
        3x4 camera-to-image projection matrix.
    box_2d
        Box as ``[[xmin, ymin], [xmax, ymax]]`` in frame pixels.
    alpha
        Local orientation of the object in radians.
    theta_ray
        Ray angle to the box center in radians.

    Returns
    -------
    np.ndarray | None
        Box center of shape (3,) in camera coordinates, or None if no corner
        assignment yielded a solvable system.
    """
    rotation = rotation_matrix(alpha + theta_ray)
    box_corners = np.array(
        [box_2d[0][0], box_2d[0][1], box_2d[1][0], box_2d[1][1]], dtype=np.float64
    )
    # Each row constrains one side of the 2D box; x sides use projected row 0,
    # y sides row 1.
    rows = (0, 1, 0, 1)

    best_loc: np.ndarray | None = None
    best_error = 1e9
    for constraint in _corner_constraints(dimension, alpha):
        a = np.zeros((4, 3))
        b = np.zeros(4)
        for row, (corner, index) in enumerate(zip(constraint, rows, strict=True)):
            m = np.eye(4)
            m[:3, 3] = rotation @ corner
            m = proj_matrix @ m
            a[row] = m[index, :3] - box_corners[row] * m[2, :3]
            b[row] = box_corners[row] * m[2, 3] - m[index, 3]

        loc, residuals, _, _ = np.linalg.lstsq(a, b, rcond=None)
        # An empty residual means the system was rank-deficient; skip it.
        if residuals.size == 0:
            continue
        if residuals[0] < best_error:
            best_error = float(residuals[0])
            best_loc = loc

    return best_loc
