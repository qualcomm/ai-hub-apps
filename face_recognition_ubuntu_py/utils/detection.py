# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np

import utils.constants as C

# Box coordinates in (x1, y1, x2, y2) order, absolute pixels in the full frame.
Box = tuple[int, int, int, int]


def load_cascade() -> cv2.CascadeClassifier:
    """
    Load the OpenCV Haar frontal-face cascade bundled with opencv-python.

    Returns
    -------
    cv2.CascadeClassifier
        A ready-to-use frontal-face detector.
    """
    cascade_path = cv2.data.haarcascades + C.HAAR_CASCADE_FILENAME
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return cascade


def detect_faces(
    rgb_frame: np.ndarray,
    cascade: cv2.CascadeClassifier,
    *,
    scale_factor: float = C.DETECTION_SCALE_FACTOR,
    min_neighbors: int = C.DETECTION_MIN_NEIGHBORS,
    min_size: tuple[int, int] = C.DETECTION_MIN_SIZE,
    max_width: int = C.DETECTION_MAX_WIDTH,
) -> list[Box]:
    """
    Detect frontal faces in an RGB frame with the Haar cascade.

    Detection runs on a grayscale copy downscaled so its width is at most
    ``max_width`` (the main cost lever for large frames); the returned boxes are
    mapped back to full-frame coordinates and clipped to the frame bounds.

    Parameters
    ----------
    rgb_frame
        RGB image of shape [H, W, 3], dtype uint8.
    cascade
        Loaded Haar cascade from :func:`load_cascade`.
    scale_factor
        Image-pyramid scale step passed to ``detectMultiScale``.
    min_neighbors
        Neighbor overlap required to keep a detection.
    min_size
        Smallest face (width, height), in detection-resolution pixels, to report.
    max_width
        Maximum width of the frame used for detection; wider frames are
        downscaled before detection.

    Returns
    -------
    list[Box]
        Detected face boxes as (x1, y1, x2, y2) tuples in full-frame pixels.
    """
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]

    scale = 1.0
    if width > max_width:
        scale = max_width / width
        gray = cv2.resize(gray, (max_width, max(1, round(height * scale))))

    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    inv_scale = 1.0 / scale
    boxes: list[Box] = []
    for x, y, box_w, box_h in detections:
        x1 = int(np.clip(round(x * inv_scale), 0, width))
        y1 = int(np.clip(round(y * inv_scale), 0, height))
        x2 = int(np.clip(round((x + box_w) * inv_scale), 0, width))
        y2 = int(np.clip(round((y + box_h) * inv_scale), 0, height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def largest_box(boxes: list[Box]) -> Box:
    """
    Return the box with the largest area.

    Parameters
    ----------
    boxes
        Non-empty list of (x1, y1, x2, y2) boxes.

    Returns
    -------
    Box
        The box covering the greatest pixel area.
    """
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def crop_face(
    rgb_frame: np.ndarray, box: Box, margin: float = C.BOX_MARGIN
) -> np.ndarray:
    """
    Crop a face from the frame, expanding the box by a margin on every side.

    Parameters
    ----------
    rgb_frame
        RGB image of shape [H, W, 3], dtype uint8.
    box
        Face box (x1, y1, x2, y2) in full-frame pixels.
    margin
        Fraction of the box width/height to add on each side before cropping.
        The expanded box is clipped to the frame bounds.

    Returns
    -------
    np.ndarray
        The cropped RGB face region.
    """
    height, width = rgb_frame.shape[:2]
    x1, y1, x2, y2 = box
    dx = round((x2 - x1) * margin)
    dy = round((y2 - y1) * margin)
    cx1 = int(np.clip(x1 - dx, 0, width))
    cy1 = int(np.clip(y1 - dy, 0, height))
    cx2 = int(np.clip(x2 + dx, 0, width))
    cy2 = int(np.clip(y2 + dy, 0, height))
    return rgb_frame[cy1:cy2, cx1:cx2]
