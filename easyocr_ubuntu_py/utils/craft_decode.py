# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""CRAFT detector score-map decoding.

Adapted from EasyOCR v1.7.2 `easyocr/craft_utils.py` (getDetBoxes_core),
Copyright (c) 2019-present NAVER Corp. (MIT); rectangular-box path only.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def get_det_boxes(
    textmap: np.ndarray,
    linkmap: np.ndarray,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Decode CRAFT text/link score maps into word-level quadrilateral boxes.

    Parameters
    ----------
    textmap
        Character-region score map, shape [H, W], float.
    linkmap
        Affinity (link) score map, shape [H, W], float.
    text_threshold
        Minimum peak region score for a component to be kept.
    link_threshold
        Threshold for binarizing the link map.
    low_text
        Threshold for binarizing the text map.

    Returns
    -------
    boxes : list[np.ndarray]
        Boxes as clockwise 4-corner arrays of shape [4, 2] (x, y) in score-map
        pixel coordinates.
    scores : list[float]
        Each box's peak character-region score (same order as `boxes`), used to
        rank boxes by detector confidence.
    """
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )
    # Independent of k, so compute once rather than every loop iteration.
    link_only = np.logical_and(link_score == 1, text_score == 0)

    det: list[np.ndarray] = []
    scores: list[float] = []
    for k in range(1, n_labels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        peak = float(np.max(textmap[labels == k]))
        if peak < text_threshold:
            continue

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        sx = max(sx, 0)
        sy = max(sy, 0)
        ex = min(ex, img_w)
        ey = min(ey, img_h)

        # Make the segmentation map local to this component's (padded) bounding
        # box instead of the full frame: `labels == k` pixels can only occur
        # inside [y:y+h, x:x+w], a subset of [sy:ey, sx:ex].
        segmap = np.zeros((ey - sy, ex - sx), dtype=np.uint8)
        segmap[labels[sy:ey, sx:ex] == k] = 255
        segmap[link_only[sy:ey, sx:ex]] = 0  # remove link area
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap = cv2.dilate(segmap, kernel)

        # make box
        np_contours = (
            np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
            .transpose()
            .reshape(-1, 2)
        )
        np_contours += [sx, sy]
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            left, right = min(np_contours[:, 0]), max(np_contours[:, 0])
            top, bottom = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array(
                [[left, top], [right, top], [right, bottom], [left, bottom]],
                dtype=np.float32,
            )

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        scores.append(peak)

    return det, scores
