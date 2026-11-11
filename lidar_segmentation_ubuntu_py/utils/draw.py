# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np

import utils.constants as C

# 256 entries so coloring can run through cv2.LUT; unused rows stay black.
_PALETTE_LUT = np.zeros((1, 256, 3), dtype=np.uint8)
_PALETTE_LUT[0, : C.NUM_CLASSES] = np.array(C.CLASS_COLORS, dtype=np.uint8)

_LEGEND_ROW_HEIGHT = 24
_LEGEND_SWATCH = 14
_LEGEND_MARGIN = 12
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_TEXT_COLOR = (230, 230, 230)


def colorize_range_view(class_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Color the predicted range image and blank the pixels with no return.

    Parameters
    ----------
    class_map
        Class-id range image of shape (height, width), dtype uint8.
    mask
        Occupancy mask of the same shape; False where no point projected.

    Returns
    -------
    np.ndarray
        RGB image of shape (C.RANGE_VIEW_HEIGHT_PX, C.RANGE_VIEW_WIDTH_PX, 3),
        dtype uint8.
    """
    colored = cv2.LUT(cv2.cvtColor(class_map, cv2.COLOR_GRAY2RGB), _PALETTE_LUT)
    colored[~mask] = 0
    return cv2.resize(
        colored,
        (C.RANGE_VIEW_WIDTH_PX, C.RANGE_VIEW_HEIGHT_PX),
        interpolation=cv2.INTER_NEAREST,
    )


def render_bev(points: np.ndarray, point_labels: np.ndarray) -> np.ndarray:
    """Rasterize the labeled point cloud into a top-down bird's-eye view.

    The sensor sits at the image center with +x (forward) pointing up and +y
    (left) pointing left. Points are drawn in ascending height so that taller
    structures win over the ground plane they stand on.

    Parameters
    ----------
    points
        Point coordinates of shape [num_points, 3], float32.
    point_labels
        Per-point class ids of shape [num_points], dtype uint8.

    Returns
    -------
    np.ndarray
        RGB image of shape (C.BEV_SIZE_PX, C.BEV_SIZE_PX, 3), dtype uint8.
    """
    scale = C.BEV_SIZE_PX / (2.0 * C.BEV_RANGE_M)
    column = (C.BEV_RANGE_M - points[:, 1]) * scale
    row = (C.BEV_RANGE_M - points[:, 0]) * scale

    keep = np.flatnonzero(
        (column >= 0)
        & (column < C.BEV_SIZE_PX)
        & (row >= 0)
        & (row < C.BEV_SIZE_PX)
        & (point_labels != 0)
    )
    order = keep[np.argsort(points[keep, 2])]

    class_grid = np.zeros((C.BEV_SIZE_PX, C.BEV_SIZE_PX), dtype=np.uint8)
    class_grid[row[order].astype(np.int32), column[order].astype(np.int32)] = (
        point_labels[order]
    )
    # Thicken before coloring, so a grown pixel keeps a real class color.
    class_grid = cv2.dilate(class_grid, np.ones((2, 2), dtype=np.uint8))
    return cv2.LUT(cv2.cvtColor(class_grid, cv2.COLOR_GRAY2RGB), _PALETTE_LUT)


def build_legend(height: int, width: int) -> np.ndarray:
    """Render the static class-color key.

    Parameters
    ----------
    height
        Height of the legend panel in pixels.
    width
        Width of the legend panel in pixels.

    Returns
    -------
    np.ndarray
        RGB image of shape (height, width, 3), dtype uint8.
    """
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    for slot, class_id in enumerate(range(1, C.NUM_CLASSES)):
        top = _LEGEND_MARGIN + slot * _LEGEND_ROW_HEIGHT
        if top + _LEGEND_SWATCH >= height:
            break
        color = tuple(int(c) for c in C.CLASS_COLORS[class_id])
        cv2.rectangle(
            legend,
            (_LEGEND_MARGIN, top),
            (_LEGEND_MARGIN + _LEGEND_SWATCH, top + _LEGEND_SWATCH),
            color,
            thickness=-1,
        )
        cv2.putText(
            legend,
            C.CLASS_NAMES[class_id],
            (_LEGEND_MARGIN * 2 + _LEGEND_SWATCH, top + _LEGEND_SWATCH),
            _FONT,
            _FONT_SCALE,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return legend


def compose_view(
    range_view: np.ndarray, bev: np.ndarray, legend: np.ndarray
) -> np.ndarray:
    """Stack the range image over the bird's-eye view and its legend.

    Parameters
    ----------
    range_view
        Colored range image spanning the full output width.
    bev
        Square bird's-eye view.
    legend
        Class-color key, the same height as the bird's-eye view.

    Returns
    -------
    np.ndarray
        RGB image of shape (range_view height + bev height, range_view width, 3),
        dtype uint8.
    """
    return np.vstack([range_view, np.hstack([bev, legend])])
