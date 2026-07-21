# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np


def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 10,
) -> None:
    """
    Draw the given points on the frame.

    Parameters
    ----------
    frame
        Numpy array representing RGB image with shape [H, W, 3] and type np.uint8.
    points
        Numpy array of shape [N, 2] where layout is [x1, y1] [x2, y2], ...
    color
        Color of drawn points (RGB)
    size
        Size of drawn points
    """
    assert points.ndim == 2 and points.shape[1] == 2
    assert isinstance(size, int) or len(size) == len(points)
    cv_keypoints = []
    for i, (x, y) in enumerate(points):
        curr_size = size if isinstance(size, int) else size[i]
        cv_keypoints.append(cv2.KeyPoint(int(x), int(y), curr_size))

    cv2.drawKeypoints(
        frame,
        cv_keypoints,
        outImage=frame,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_connections(
    frame: np.ndarray,
    points: np.ndarray,
    connections: list[tuple[int, int]],
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 1,
) -> None:
    """
    Draw connecting lines between the given points on the frame.

    Parameters
    ----------
    frame
        Numpy array representing RGB image with shape [H, W, 3] and type np.uint8.
    points
        Numpy array of shape [N, 2] where layout is [x1, y1] [x2, y2], ...
    connections
        List of points that should be connected by a line.
        Format is [(src point index, dst point index), ...]
    color
        Color of drawn points (RGB)
    size
        Size of drawn connection lines
    """
    point_pairs: list[tuple[tuple[int, int], tuple[int, int]]] | np.ndarray
    assert connections is not None
    point_pairs = [
        (
            (int(points[i][0]), int(points[i][1])),
            (int(points[j][0]), int(points[j][1])),
        )
        for (i, j) in connections
    ]
    cv2.polylines(
        frame,
        np.asarray(point_pairs, dtype=np.int64),
        isClosed=False,
        color=color,
        thickness=size,
    )


def draw_box_from_xyxy(
    frame: np.ndarray,
    top_left: np.ndarray | tuple[int, int],
    bottom_right: np.ndarray | tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
    text: str | None = None,
) -> None:
    """
    Draw a box using the provided top left / bottom right points to compute the box.

    Parameters
    ----------
    frame
        Numpy array representing RGB image with shape [H, W, 3] and type np.uint8.
    top_left
        Top-left coordinate.
    bottom_right
        Bottom-right coordinate.
    color
        Color of drawn points and connection lines (RGB)
    size
        Size of drawn points and connection lines RGB channel layout
    text
        Overlay text at the top of the box.
    """
    if not isinstance(top_left, tuple):
        top_left = (int(top_left[0].item()), int(top_left[1].item()))
    if not isinstance(bottom_right, tuple):
        bottom_right = (int(bottom_right[0].item()), int(bottom_right[1].item()))
    cv2.rectangle(frame, top_left, bottom_right, color, size)
    if text is not None:
        cv2.putText(
            frame,
            text,
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            size,
        )
