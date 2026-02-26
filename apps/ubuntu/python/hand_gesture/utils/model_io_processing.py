# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import List

import numpy as np
import utils.constants as C
from utils.bbox_processing import (
    apply_directional_box_offset,
    box_xyxy_to_xywh,
    compute_box_corners_with_rotation,
)
from utils.image_processing import compute_vector_rotation


def compute_object_roi(
    batched_selected_boxes: List[np.ndarray],
    batched_selected_keypoints: List[np.ndarray],
) -> List[np.ndarray]:
    """
    From the provided bounding boxes and keypoints, compute the region of interest (ROI) that should be used
    as input to the landmark detection model.

    Parameters
    ----------
    batched_selected_boxes
        Selected object bounding box coordinates. Empty array if batch had no bounding boxes with a score above the threshold.
        Shape of each list element is [num_selected_boxes, 2, 2].
            Layout is:
                [[box_x1, box_y1],
                 [box_x2, box_y2]]

    batched_selected_keypoints
        Selected object bounding box keypoints. Empty array if batch had no bounding boxes with a score above the threshold.
        Shape of each list element is [num_selected_boxes, # of keypoints, 2].
            Layout is:
                [[keypoint_0_x, keypoint_0_y],
                 ...,
                 [keypoint_max_x, keypoint_max_y]]

    Returns
    -------
    batched_roi_4corners
        Selected object "region of interest" (region used as input to the landmark detector) corner coordinates.
        Empty array if batch had no bounding boxes with a score above the threshold.
        Shape of each list element is [num_selected_boxes, 4, 2], where the 2 corresponds to (x, y).
        The order of points is (top left, bottom left, top right, bottom right).

    Notes
    -----
    This NumPy port assumes NumPy equivalents for the referenced helpers and constants:
      - compute_vector_rotation(...)
      - box_xyxy_to_xywh(...)
      - apply_directional_box_offset(...)
      - compute_box_corners_with_rotation(...)
      - keypoint_rotation_vec_start_idx, keypoint_rotation_vec_end_idx
      - rotation_offset_rads, detect_box_offset_xy, detect_box_scale

    The behavior for the "empty" case mirrors the original (returns a 1-D empty array).
    """
    batched_selected_roi: List[np.ndarray] = []

    for boxes, keypoints in zip(
        batched_selected_boxes, batched_selected_keypoints, strict=False
    ):
        boxes = np.asarray(boxes)
        keypoints = np.asarray(keypoints)

        if boxes.size == 0 or keypoints.size == 0:
            batched_selected_roi.append(np.array([]))
            continue

        # Compute bounding box center and rotation
        theta = compute_vector_rotation(
            keypoints[:, C.KEYPOINT_ROTATION_VEC_START_IDX, ...],
            keypoints[:, C.KEYPOINT_ROTATION_VEC_END_IDX, ...],
            C.ROTATION_OFFSET_RADS,
        )

        selected_boxes_cwh = box_xyxy_to_xywh(boxes)  # expected shape [N, 2, 2]
        # Copy to ensure we can safely mutate without affecting any shared backing arrays
        xc = selected_boxes_cwh[..., 0, 0].copy()
        yc = selected_boxes_cwh[..., 0, 1].copy()
        w = selected_boxes_cwh[..., 1, 0].copy()
        h = selected_boxes_cwh[..., 1, 1].copy()

        # Move the box to better center the object (in-place update of xc, yc expected)
        apply_directional_box_offset(
            C.DETECT_BOX_OFFSET_XY * w,
            keypoints[..., C.KEYPOINT_ROTATION_VEC_START_IDX, :],
            keypoints[..., C.KEYPOINT_ROTATION_VEC_END_IDX, :],
            xc,
            yc,
        )

        # Enlarge the box
        w *= C.DETECT_BOX_SCALE
        h *= C.DETECT_BOX_SCALE

        # Compute 4 corner points of the rotated ROI for each selected box
        roi_4corners = compute_box_corners_with_rotation(xc, yc, w, h, theta)
        batched_selected_roi.append(roi_4corners)

    return batched_selected_roi


def preprocess_hand_x64(
    pts: np.ndarray, handedness: np.ndarray, mirror: bool = False
) -> np.ndarray:
    """
    Normalize hand landmarks, flatten (63), and concatenate handedness (1) → x64.

    Parameters
    ----------
    pts : np.ndarray
        Landmark points, shape [N, 21, 3].

    handedness : np.ndarray
        Handedness flags, shape [N, 1]. Typically 0=left, 1=right.

    mirror : bool
        If True, mirror across X-axis (flip X) and invert handedness.

    Returns
    -------
    x64 : np.ndarray
        Preprocessed features of shape [N, 64]: 63 normalized landmark coords + 1 handedness.
    """
    # Ensure arrays
    pts = np.asarray(pts)
    handedness = np.asarray(handedness)

    if mirror:
        # Flip X (assuming (x, y, z) ordering); keep shape (1,1,3) for broadcasting
        x_mirror = np.array([-1.0, 1.0, 1.0], dtype=pts.dtype).reshape(1, 1, 3)
        pts = pts * x_mirror
        # Invert handedness (assumes 0/1 semantics or probabilities in [0,1])
        handedness = 1.0 - handedness

    # Fixed normalization configuration
    # stable anatomical anchors: wrist (0), MCPs (1, 5, 9, 13, 17)
    center_idx = np.array([0, 1, 5, 9, 13, 17], dtype=np.int64)
    epsilon = 1e-5  # avoid divide-by-zero

    # Compute center from selected landmarks -> (N, 1, 3)
    center = pts[:, center_idx, :].mean(axis=1, keepdims=True)

    # Translate points so center is at origin
    normed = pts - center

    # Compute scale based on max range in X or Y (per sample)
    x = normed[..., 0]  # (N, 21)
    y = normed[..., 1]  # (N, 21)
    range_x = x.max(axis=1) - x.min(axis=1)  # (N,)
    range_y = y.max(axis=1) - y.min(axis=1)  # (N,)
    scale = np.maximum(range_x, range_y).reshape(-1, 1, 1) + epsilon  # (N,1,1)

    # Normalize and flatten landmarks to 63 features
    pts_n = normed / scale
    flat = pts_n.reshape(pts_n.shape[0], 63)

    # Append handedness scalar (cast to float32 for consistency)
    x64 = np.concatenate([flat, handedness.reshape(-1, 1).astype(np.float32)], axis=1)
    return x64
