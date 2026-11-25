# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Group CRAFT boxes into text lines and warp free-form boxes.

Adapted from EasyOCR v1.7.2 `easyocr/utils.py` (group_text_box,
four_point_transform, diff), Copyright (c) JaidedAI (Apache-2.0).
"""

from __future__ import annotations

import cv2
import numpy as np

# Box type aliases.
# Horizontal box: (x_min, x_max, y_min, y_max)
BoxXXYY = tuple[int, int, int, int]
# Free box: ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
BoxCorners = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


def diff(input_list: list[float]) -> float:
    return max(input_list) - min(input_list)


def four_point_transform(image: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """Perspective-warp the quadrilateral `rect` out of `image` into a rectangle."""
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def group_text_box(
    polys: list[np.ndarray],
    slope_ths: float = 0.1,
    ycenter_ths: float = 0.5,
    height_ths: float = 0.5,
    width_ths: float = 1.0,
    add_margin: float = 0.05,
    sort_output: bool = True,
) -> tuple[list[list[int]], list[list[list[float]]]]:
    """
    Merge word boxes into text-line boxes.

    Parameters
    ----------
    polys
        List of flat 8-element boxes [x1,y1,x2,y2,x3,y3,x4,y4] (top-left, top-right,
        bottom-right, bottom-left) in original-image pixel coordinates.
    slope_ths
        Maximum box slope (rise/run) to treat a box as horizontal rather than slanted.
    ycenter_ths
        Max y-center deviation (as a multiple of mean box height) to combine two
        horizontal boxes onto the same line.
    height_ths
        Max height difference (as a fraction of mean height) to merge two boxes on
        the same line.
    width_ths
        Max horizontal gap (as a multiple of box height) to merge two boxes on the
        same line.
    add_margin
        Margin added around merged boxes, as a fraction of the box's shorter side.
    sort_output
        If True, sort horizontal boxes by y-center before merging.

    Returns
    -------
    horizontal_list : list[list[int]]
        Axis-aligned line boxes [x_min, x_max, y_min, y_max].
    free_list : list[list[list[float]]]
        Slanted line boxes as 4 corner points [[x,y], ...].
    """
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    for poly in polys:
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            horizontal_list.append(
                [x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min]
            )
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])

            margin = int(1.44 * add_margin * min(width, height))

            theta13 = abs(
                np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4])))
            )
            theta24 = abs(
                np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6])))
            )
            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin

            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box: list = []
    b_height: list = []
    b_ycenter: list = []
    for box in horizontal_list:
        if len(new_box) == 0:
            b_height = [box[5]]
            b_ycenter = [box[4]]
            new_box.append(box)
        # comparable height and comparable y_center level up to ths*height
        elif abs(np.mean(b_ycenter) - box[4]) < ycenter_ths * np.mean(b_height):
            b_height.append(box[5])
            b_ycenter.append(box[4])
            new_box.append(box)
        else:
            b_height = [box[5]]
            b_ycenter = [box[4]]
            combined_list.append(new_box)
            new_box = [box]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1:  # one box per line
            box = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            merged_list.append(
                [box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]
            )
        else:  # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box: list = []
            new_box = []
            x_max = 0
            for box in boxes:
                if len(new_box) == 0:
                    b_height = [box[5]]
                    x_max = box[1]
                    new_box.append(box)
                elif (
                    abs(np.mean(b_height) - box[5]) < height_ths * np.mean(b_height)
                ) and ((box[0] - x_max) < width_ths * (box[3] - box[2])):  # merge boxes
                    b_height.append(box[5])
                    x_max = box[1]
                    new_box.append(box)
                else:
                    b_height = [box[5]]
                    x_max = box[1]
                    merged_box.append(new_box)
                    new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # adjacent box in same line
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append(
                        [x_min - margin, x_max + margin, y_min - margin, y_max + margin]
                    )
                else:  # non adjacent box in same line
                    box = mbox[0]

                    box_width = box[1] - box[0]
                    box_height = box[3] - box[2]
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append(
                        [
                            box[0] - margin,
                            box[1] + margin,
                            box[2] - margin,
                            box[3] + margin,
                        ]
                    )
    return merged_list, free_list
