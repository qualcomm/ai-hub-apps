# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import List, Tuple

import cv2
import numpy as np


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) between a single box and an array of boxes.

    Parameters
    ----------
    box : np.ndarray
        The reference box in the format ``(x1, y1, x2, y2)``.
    boxes : np.ndarray
        An array of boxes, each in the format ``(x1, y1, x2, y2)``.

    Returns
    -------
    ndarray of shape (N,)
        IoU values for each box in `boxes` with respect to `box`.
    """

    # Intersection coordinates
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    return intersection / np.maximum(union, 1e-10)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:

    """
    Perform standard Non-Maximum Suppression (NMS) on a set of bounding boxes.

    Parameters
    ----------
    boxes : np.ndarray of shape (N, 4)
        Bounding boxes in the format ``(x1, y1, x2, y2)``.
    scores : np.ndarray of shape (N,)
        Confidence scores associated with each box.
    iou_threshold : float
        IoU threshold used to determine whether boxes overlap too much.

    Returns
    -------
    np.ndarray of shape (M,)
        Indices of the boxes that are kept after NMS, where ``M ≤ N``.

    """
    # Sort by score descending
    order = np.argsort(scores)[::-1]

    keep = []
    while len(order) > 0:
        # Pick the box with highest score
        idx = order[0]
        keep.append(idx)

        if len(order) == 1:
            break

        # Compute IoU with remaining boxes
        remaining = order[1:]
        ious = _compute_iou(boxes[idx], boxes[remaining])

        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        order = remaining[mask]

    return np.array(keep, dtype=np.int64)


def _batched_nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:

    """
    Apply Non-Maximum Suppression (NMS) independently for each class.

    Parameters
    ----------
    boxes : np.ndarray of shape (N, 4)
        Bounding boxes in the format ``(x1, y1, x2, y2)``.
    scores : np.ndarray of shape (N,)
        Confidence scores for each bounding box.
    class_indices : np.ndarray of shape (N,)
        Class ID for each box, used to group boxes before applying NMS.
    iou_threshold : float
        IoU threshold used to determine suppression.

    Returns
    -------
    np.ndarray of shape (M,)
        Indices of boxes kept after per-class NMS, where ``M ≤ N``.
    """

    unique_classes = np.unique(class_indices)
    keep_all = []

    for cls in unique_classes:
        cls_mask = class_indices == cls
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        cls_keep = _nms(cls_boxes, cls_scores, iou_threshold)
        keep_all.append(cls_indices[cls_keep])

    if len(keep_all) == 0:
        return np.array([], dtype=np.int64)

    keep = np.concatenate(keep_all)
    # Sort by score to maintain consistent ordering
    keep = keep[np.argsort(scores[keep])[::-1]]
    return keep


def batched_nms(
    iou_threshold: float,
    score_threshold: float | None,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray | None = None,
) -> tuple[list[np.ndarray], ...]:
    """
    Non maximum suppression over several batches.

    Parameters
    ----------
    iou_threshold : float
        Intersection over union (IoU) threshold
    score_threshold : float or None
        Score threshold (throw away any boxes with scores under this threshold)
    boxes : np.ndarray of shape (B, N, 4)
        Boxes to run NMS on. Shape is [B, N, 4], B == batch, N == num boxes,
        and 4 == (x1, y1, x2, y2)
    scores : np.ndarray of shape (B, N)
        Scores for each box. Shape is [B, N], range is [0:1]
    class_indices : np.ndarray of shape (B, N), optional
        Class for each box. Shape is [B, N].
        If set, NMS is applied per-class rather than globally.

    Returns
    -------
    boxes_out : List[np.ndarray]
        Output boxes. This is list of arrays; one array per batch.
        Each array is shape [S, 4], where S == number of selected boxes,
        and 4 == (x1, y1, x2, y2)
    scores_out : List[np.ndarray]
        Output scores. This is list of arrays; one array per batch.
        Each array is shape [S], where S == number of selected boxes.
    class_indices_out : List[np.ndarray]
        Output classes. This is list of arrays; one array per batch.
        Each array is shape [S], where S == number of selected boxes.
        if class_indices is None (default), this is an empty list.
    """
    scores_out: list[np.ndarray] = []
    boxes_out: list[np.ndarray] = []
    class_indices_out: list[np.ndarray] = []

    for batch_idx in range(boxes.shape[0]):
        # Index to current batch
        batch_scores = scores[batch_idx]
        batch_boxes = boxes[batch_idx]
        batch_class_indices = (
            class_indices[batch_idx] if class_indices is not None else None
        )

        # Clip outputs to valid scores
        if score_threshold is not None:
            scores_idx = np.where(scores[batch_idx] >= score_threshold)[0]
            batch_scores = batch_scores[scores_idx]
            batch_boxes = batch_boxes[scores_idx]
            batch_class_indices = (
                batch_class_indices[scores_idx]
                if batch_class_indices is not None
                else None
            )

        if len(batch_scores) > 0:
            # Apply NMS
            if batch_class_indices is not None:
                # class dependent
                nms_indices = _batched_nms_numpy(
                    batch_boxes[..., :4],
                    batch_scores,
                    batch_class_indices,
                    iou_threshold,
                )
            else:
                # class agnostic
                nms_indices = _nms(batch_boxes[..., :4], batch_scores, iou_threshold)

            # Apply NMS indices
            batch_boxes = batch_boxes[nms_indices]
            batch_scores = batch_scores[nms_indices]
            batch_class_indices = (
                batch_class_indices[nms_indices]
                if batch_class_indices is not None
                else None
            )

        # Append to outputs
        boxes_out.append(batch_boxes)
        scores_out.append(batch_scores)
        if batch_class_indices is not None:
            class_indices_out.append(batch_class_indices)

    return boxes_out, scores_out, class_indices_out


def box_xywh_to_xyxy(box_cwh: np.ndarray, flat_boxes: bool = False) -> np.ndarray:
    """
    Convert center (xc, yc), width (w), height (h) to (x0, y0, x1, y1).

    Parameters
    ----------
    box_cwh : np.ndarray
        Bounding boxes.
        If flat_boxes:
            Shape is [..., 4] with layout [xc, yc, w, h]
        else:
            Shape is [..., 2, 2] with layout [[xc, yc], [w, h]]
    flat_boxes : bool
        Whether input is in flat layout.

    Returns
    -------
    box_xyxy : np.ndarray
        If flat_boxes:
            Shape [..., 4] with layout [x0, y0, x1, y1]
        else:
            Shape [..., 2, 2] with layout [[x0, y0], [x1, y1]]
    """
    box_cwh = np.asarray(box_cwh)

    if flat_boxes:
        cx = box_cwh[..., 0]
        cy = box_cwh[..., 1]
        w_2 = box_cwh[..., 2] * 0.5
        h_2 = box_cwh[..., 3] * 0.5

        x0 = cx - w_2
        y0 = cy - h_2
        x1 = cx + w_2
        y1 = cy + h_2
        return np.stack((x0, y0, x1, y1), axis=-1)

    # Structured layout: [[xc, yc], [w, h]]
    x_center = box_cwh[..., 0, 0]
    y_center = box_cwh[..., 0, 1]
    w = box_cwh[..., 1, 0]
    h = box_cwh[..., 1, 1]

    out = box_cwh.copy()
    out[..., 0, 0] = x_center - w / 2.0  # x0
    out[..., 0, 1] = y_center - h / 2.0  # y0
    out[..., 1, 0] = x_center + w / 2.0  # x1
    out[..., 1, 1] = y_center + h / 2.0  # y1

    return out


def box_xyxy_to_xywh(box_xy: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from (x0, y0, x1, y1)
    to center-width-height format.

    Parameters
    ----------
    box_xy : np.ndarray
        Bounding box tensor shaped [B, 2, 2]
        where:
            box_xy[..., 0, :] = (x0, y0)
            box_xy[..., 1, :] = (x1, y1)

    Returns
    -------
    box_cwh : np.ndarray
        Bounding box shaped [B, 2, 2] with:
            [0, :] = (xc, yc)
            [1, :] = (w, h)
    """
    box_xy = np.asarray(box_xy)
    out = box_xy.copy()

    x0 = box_xy[..., 0, 0]
    y0 = box_xy[..., 0, 1]
    x1 = box_xy[..., 1, 0]
    y1 = box_xy[..., 1, 1]

    w = x1 - x0
    h = y1 - y0
    xc = x0 + w / 2
    yc = y0 + h / 2

    out[..., 1, 0] = w
    out[..., 1, 1] = h
    out[..., 0, 0] = xc
    out[..., 0, 1] = yc

    return out


def apply_directional_box_offset(
    offset: float | np.ndarray,
    vec_start: np.ndarray,
    vec_end: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
) -> None:
    """
    Offset the bounding box defined by [xc, yc] by a pre-determined length.
    The offset is applied along the direction from vec_start -> vec_end.

    Parameters
    ----------
    offset : float or np.ndarray
        Offset magnitude (absolute units). Can be scalar or array broadcastable to [B].
    vec_start : np.ndarray
        Starting point of the vector. Shape [B, 2] where 2 == (x, y).
    vec_end : np.ndarray
        Ending point of the vector. Shape [B, 2] where 2 == (x, y).
    xc : np.ndarray
        x center(s) of box(es). Modified in-place.
    yc : np.ndarray
        y center(s) of box(es). Modified in-place.

    Returns
    -------
    None
        `xc` and `yc` are updated in place.
    """
    vec_start = np.asarray(vec_start)
    vec_end = np.asarray(vec_end)

    # Vector components
    xlen = vec_end[..., 0] - vec_start[..., 0]
    ylen = vec_end[..., 1] - vec_start[..., 1]

    # Vector length (avoid division by zero with small epsilon)
    vec_len = np.sqrt(np.square(xlen) + np.square(ylen))
    eps = 1e-12
    safe_len = np.maximum(vec_len, eps)

    # Unit direction * offset
    dx = offset * (xlen / safe_len)
    dy = offset * (ylen / safe_len)

    # In-place updates
    xc += dx
    yc += dy


def compute_box_corners_with_rotation(
    xc: np.ndarray,
    yc: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    From the provided information, compute the (x, y) coordinates of the box's corners.

    Parameters
    ----------
    xc : np.ndarray
        Center of box (x). Shape [B]
    yc : np.ndarray
        Center of box (y). Shape [B]
    w : np.ndarray
        Width of box. Shape [B]
    h : np.ndarray
        Height of box. Shape [B]
    theta : np.ndarray
        Rotation of box (in radians). Shape [B]

    Returns
    -------
    corners : np.ndarray
        Computed corners. Shape [B, 4, 2], where the last dim is (x, y).
        Corner order is (top-left, bottom-left, top-right, bottom-right).
    """
    # Ensure arrays
    xc = np.asarray(xc)
    yc = np.asarray(yc)
    w = np.asarray(w)
    h = np.asarray(h)
    theta = np.asarray(theta)

    batch_size = xc.shape[0]

    # Construct unit square in a fixed corner order: TL, BL, TR, BR
    # Shape before repeat: [2, 4], where rows are (x; y)
    base = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=np.float32)

    # Repeat across batch -> [B, 2, 4]
    points = np.broadcast_to(base, (batch_size, *base.shape)).copy()

    # Scale to half-width and half-height: [B, 2] -> unsqueeze to [B, 2, 1] for broadcast
    half_wh = np.stack((w / 2.0, h / 2.0), axis=-1, dtype=np.float32)[
        :, :, None
    ]  # [B, 2, 1]
    points = points * half_wh  # [B, 2, 4]

    # Rotation matrices per item: R = [[cos, -sin], [sin, cos]]  -> [B, 2, 2]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.stack(
        (
            np.stack((cos_t, -sin_t), axis=1),
            np.stack((sin_t, cos_t), axis=1),
        ),
        axis=1,
    )  # [B, 2, 2]

    # Apply rotation: [B, 2, 2] @ [B, 2, 4] -> [B, 2, 4]
    points = R @ points

    # Translate by (xc, yc): stack to [B, 2, 1] for broadcast add
    centers = np.stack((xc, yc), axis=1)[:, :, None]  # [B, 2, 1]
    points = points + centers  # [B, 2, 4]

    # Return as [B, 4, 2] with last dim = (x, y)
    return np.swapaxes(points, -1, -2)  # [B, 4, 2]


def compute_box_affine_crop_resize_matrix(
    box_corners: np.ndarray, output_image_size: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Compute the affine transform matrices required to crop, rescale, and pad the
    rotated box defined by the input corners to fit into an output image size
    without warping.

    Parameters
    ----------
    box_corners : np.ndarray
        Bounding box corners to map *from*. Shape [B, K, 2], where:
          - B = batch size
          - K >= 3 corners (expected order: top-left, bottom-left, top-right, (optional bottom-right))
          - last dim = (x, y)
        If K > 3, only the first 3 corners are used (TL, BL, TR), matching the original logic.

    output_image_size : Tuple[int, int]
        Output (width, height) to which the box is mapped.
        Note: This function expects a tuple in the order (W, H).

    Returns
    -------
    affines : List[np.ndarray]
        List of affine matrices, each of shape (2, 3), one per batch element.
    """
    # Unpack target width/height;
    out_w, out_h = output_image_size

    # Destination triangle (target positions) in the output image:
    # top-left -> (0, 0)
    # bottom-left -> (0, H-1)
    # top-right -> (W-1, 0)
    network_input_points = np.array(
        [[0, 0], [0, out_h - 1], [out_w - 1, 0]], dtype=np.float32
    )

    # Ensure numpy array
    box_corners = np.asarray(box_corners)

    # Validate minimal shape
    if box_corners.ndim != 3 or box_corners.shape[-1] != 2:
        raise ValueError(
            f"`box_corners` must have shape [B, K, 2]; got {box_corners.shape}"
        )
    if box_corners.shape[1] < 3:
        raise ValueError(
            f"`box_corners` must provide at least 3 corners per item; got K={box_corners.shape[1]}"
        )

    affines: List[np.ndarray] = []
    B = box_corners.shape[0]
    for b in range(B):
        # Use only the first 3 corners (TL, BL, TR) to match original behavior
        # Ensure float32 for OpenCV
        src = box_corners[b, :3, :].astype(np.float32, copy=False)
        # Compute 2x3 affine transform mapping src -> network_input_points
        M = cv2.getAffineTransform(src, network_input_points)
        affines.append(M)

    return affines
