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
):
    """
    Draw the given points on the frame.

    Parameters
    ----------
    frame: np.ndarray
        Numpy array representing RGB image with shape [H, W, 3] and type np.uint8.

    points: np.ndarray
        Numpy array of shape [N, 2] where layout is
            [x1, y1] [x2, y2], ...

    color: tuple[int, int, int]
        Color of drawn points (RGB)

    size: int
        Size of drawn points

    Returns
    -------
        None; modifies frame in place.
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
):
    """
    Draw connecting lines between the given points on the frame.

    Parameters
    ----------
    frame:
        Numpy array representing RGB image with shape [H, W, 3] and type np.uint8.

    points:
        Numpy array of shape [N, 2] where layout is
            [x1, y1] [x2, y2], ...

    connections:
        List of points that should be connected by a line.
        Format is [(src point index, dst point index), ...]

    color:
        Color of drawn points (RGB)

    size: int
        Size of drawn connection lines

    Returns
    -------
        None; modifies frame in place.
    """
    point_pairs: (list[tuple[tuple[int, int], tuple[int, int]]] | np.ndarray)
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
        thickness=size,  # type: ignore[call-overload]
    )


def draw_box_from_xyxy(
    frame: np.ndarray,
    top_left: np.ndarray | tuple[int, int],
    bottom_right: np.ndarray | tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
    text: str | None = None,
):
    """
    Draw a box using the provided top left / bottom right points to compute the box.

    Parameters
    ----------
    frame: np.ndarray
        Numpy array representing RGB image with shape [H, W, 3] and type np.uint8.

    top_left : np.ndarray | tuple[int, int]
        Top-left coordinate.

    bottom_right : np.ndarray | tuple[int, int]
        Bottom-right coordinate.

    color: tuple[int, int, int]
        Color of drawn points and connection lines (RGB)

    size: int
        Size of drawn points and connection lines RGB channel layout

    text: None | str
        Overlay text at the top of the box.

    Returns
    -------
        None; modifies frame in place.
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


def draw_predictions(
    NHWC_int_numpy_frames: list[np.ndarray],
    batched_selected_landmarks: list[np.ndarray],
    batched_is_right_hand: list[list[bool]],
    batched_gesture_labels: list[list[str]],
    landmark_connections: list[tuple[int, int]] | None = None,
):
    """
    Draws batched predictions

    Parameters
    ----------
    NHWC_int_numpy_frame: List[np.ndarray]
        List of Numpy array of shape [H, W, 3] in BGR (OpenCV) format.

    batched_selected_landmarks: List[np.ndarray]
        List of Numpy array of shape [B, N, D] where indexes 0, 1 across dimension 3 are x, y.

    batched_is_right_hand: List[List[bool]]
        True if the detection is a right hand, false if it's a left hand. None if no hand detected.

    batched_gesture_labels: List[str]
        List of string with resolved labels per hand.

    landmark_connections: List[Tuple[int, int]] | None
        List of connections between landmark points.
    """
    for batch_idx in range(len(NHWC_int_numpy_frames)):
        image = NHWC_int_numpy_frames[batch_idx]
        ld = batched_selected_landmarks[batch_idx]
        irh = batched_is_right_hand[batch_idx]
        gestures = batched_gesture_labels[batch_idx]
        if ld.size != 0 and len(irh) != 0:
            draw_landmarks_gesture_label(
                image, ld, irh, gestures, landmark_connections=landmark_connections
            )


def draw_landmarks_gesture_label(
    NHWC_int_numpy_frame: np.ndarray,
    landmarks: np.ndarray,
    is_right_hand: list[bool],
    gesture_labels: list[str],
    coords_normalized: bool = False,
    landmark_connections: list[tuple[int, int]] | None = None,
):
    """
    Draw landmarks, overlay 'Left/Right: <gesture>' and gesture label near each hand on the image.

    Parameters
    ----------
    NHWC_int_numpy_frame: np.ndarray
        Numpy array of shape [H, W, 3] in BGR (OpenCV) format.

    landmarks
        Numpy array of shape [M, N, D] where indexes 0, 1 across dimension 3 are x, y.

    is_right_hand: List[bool]
        List of boolean of length M.

    gesture_labels: List[str]
        List of string of length M with resolved labels per hand.

    coords_normalized
        If True, x,y are in [0, 1] and will be converted to pixel coordinates.
    """
    H, W = NHWC_int_numpy_frame.shape[:2]

    for ldm, irh, gest in zip(landmarks, is_right_hand, gesture_labels, strict=False):
        # Convert landmarks to numpy
        xy = ldm[:, [0, 1]]

        # Convert normalized coords to pixel coords if needed
        xy_px = (
            np.column_stack([xy[:, 0] * W, xy[:, 1] * H]) if coords_normalized else xy
        )

        # Draw landmark points and connections
        draw_points(NHWC_int_numpy_frame, xy_px, (0, 0, 255))
        if landmark_connections:
            draw_connections(
                NHWC_int_numpy_frame,
                xy_px,
                landmark_connections,
                (0, 255, 0),
                2,
            )

        # Compute bounding box from landmarks
        x_min, y_min = xy_px.min(axis=0).astype(int)
        x_max, y_max = xy_px.max(axis=0).astype(int)

        # Prepare label text
        handedness = "Right" if irh else "Left"
        label_text = f"{handedness}: {gest}"

        # Use helper for box + text overlay
        draw_box_from_xyxy(
            NHWC_int_numpy_frame,
            top_left=(x_min - 20, y_min - 20),
            bottom_right=(x_max + 20, y_max + 20),
            color=(255, 0, 0),  # Box color
            size=2,
            text=label_text,
        )
