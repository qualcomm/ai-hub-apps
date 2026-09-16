# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

import utils.constants as C


def traverse_to_targ_keypoint(
    edge_id: int,
    source_keypoint: np.ndarray,
    target_keypoint_id: int,
    scores: np.ndarray,
    offsets: np.ndarray,
    displacements: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Given a source keypoint and target_keypoint_id, predict the score and
    coordinates of the target keypoint.

    Parameters
    ----------
    edge_id
        Index of the edge being considered. Equivalent to the index in `POSE_CHAIN`.
    source_keypoint
        (y, x) coordinates of the keypoint.
    target_keypoint_id
        Which body part type of the 17 this keypoint is.
    scores
        See `decode_multiple_poses`.
    offsets
        See `decode_multiple_poses`.
    displacements
        See `decode_multiple_poses`.

    Returns
    -------
    score : float
        Target keypoint score.
    image_coord : np.ndarray
        Target keypoint coordinates.
    """
    height = scores.shape[1]
    width = scores.shape[2]

    source_keypoint_indices = np.clip(
        np.round(source_keypoint / C.OUTPUT_STRIDE),
        a_min=0,
        a_max=[height - 1, width - 1],
    ).astype(np.int32)

    displaced_point = (
        source_keypoint
        + displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]]
    )

    displaced_point_indices = np.clip(
        np.round(displaced_point / C.OUTPUT_STRIDE),
        a_min=0,
        a_max=[height - 1, width - 1],
    ).astype(np.int32)

    score = scores[
        target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]
    ]

    image_coord = (
        displaced_point_indices * C.OUTPUT_STRIDE
        + offsets[
            target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]
        ]
    )

    return score, image_coord


def decode_pose(
    root_score: float,
    root_id: int,
    root_image_coord: np.ndarray,
    scores: np.ndarray,
    offsets: np.ndarray,
    displacements_fwd: np.ndarray,
    displacements_bwd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get all keypoint predictions for a pose given a root keypoint with a high score.

    Parameters
    ----------
    root_score
        The confidence score of the root keypoint.
    root_id
        Which body part type of the 17 this keypoint is.
    root_image_coord
        (y, x) coordinates of the keypoint.
    scores
        See `decode_multiple_poses`.
    offsets
        See `decode_multiple_poses`.
    displacements_fwd
        See `decode_multiple_poses`.
    displacements_bwd
        See `decode_multiple_poses`.

    Returns
    -------
    instance_keypoint_scores : np.ndarray
        List of keypoint scores.
    instance_keypoint_coords : np.ndarray
        List of keypoint coordinates.
    """
    num_parts = scores.shape[0]
    num_edges = len(C.PARENT_CHILD_TUPLES)

    instance_keypoint_scores = np.zeros(num_parts)
    instance_keypoint_coords = np.zeros((num_parts, 2))
    instance_keypoint_scores[root_id] = root_score
    instance_keypoint_coords[root_id] = root_image_coord

    for edge in reversed(range(num_edges)):
        target_keypoint_id, source_keypoint_id = C.PARENT_CHILD_TUPLES[edge]
        if (
            instance_keypoint_scores[source_keypoint_id] > 0.0
            and instance_keypoint_scores[target_keypoint_id] == 0.0
        ):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores,
                offsets,
                displacements_bwd,
            )
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    for edge in range(num_edges):
        source_keypoint_id, target_keypoint_id = C.PARENT_CHILD_TUPLES[edge]
        if (
            instance_keypoint_scores[source_keypoint_id] > 0.0
            and instance_keypoint_scores[target_keypoint_id] == 0.0
        ):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores,
                offsets,
                displacements_fwd,
            )
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    return instance_keypoint_scores, instance_keypoint_coords


def within_nms_radius_fast(
    pose_coords: np.ndarray, nms_radius: float, point: np.ndarray
) -> bool:
    """
    Whether the candidate point is nearby any existing point in `pose_coords`.

    Parameters
    ----------
    pose_coords
        Numpy array of points, shape (N, 2).
    nms_radius
        The distance between two points for them to be considered nearby.
    point
        The candidate point, shape (2,).

    Returns
    -------
    within_radius : bool
        Whether the point is within the NMS radius of any existing pose coords.
    """
    if not pose_coords.shape[0]:
        return False
    return bool(np.any(np.sum((pose_coords - point) ** 2, axis=1) <= nms_radius**2))


def get_instance_score_fast(
    exist_pose_coords: np.ndarray,
    nms_radius: int,
    keypoint_scores: np.ndarray,
    keypoint_coords: np.ndarray,
) -> float:
    """
    Compute a probability that the given pose is real. Equal to the average
    confidence of each keypoint, excluding keypoints that are shared with
    existing poses.

    Parameters
    ----------
    exist_pose_coords
        Keypoint coordinates of poses that have already been found. Shape (N, 17, 2).
    nms_radius
        If two candidate keypoints for the same body part are within this distance,
        they are considered the same, and the lower confidence one discarded.
    keypoint_scores
        Keypoint scores for the new pose. Shape (17,).
    keypoint_coords
        Coordinates for the new pose. Shape (17, 2).

    Returns
    -------
    confidence_score : float
        Confidence score for the pose.
    """
    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2) > nms_radius**2
        not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return float(not_overlapped_scores / len(keypoint_scores))


def max_pool_2d(scores: np.ndarray, radius: int) -> np.ndarray:
    """
    Max pool each (h, w) plane of `scores` with a square kernel of size
    2 * radius + 1, stride 1, keeping the spatial dimensions unchanged.

    Parameters
    ----------
    scores
        See `decode_multiple_poses`. Shape [17, h, w].
    radius
        Neighborhood radius of the pooling kernel.

    Returns
    -------
    max_pooled : np.ndarray
        Element-wise local maximum of `scores`, same shape as `scores`.
    """
    kernel = 2 * radius + 1
    # -inf padding so out-of-bounds neighbors never win the max, matching the
    # behavior of torch.nn.functional.max_pool2d.
    padded = np.pad(
        scores,
        ((0, 0), (radius, radius), (radius, radius)),
        mode="constant",
        constant_values=-np.inf,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel, kernel), axis=(1, 2)
    )
    return windows.max(axis=(-2, -1))


def build_part_with_score(
    score_threshold: float, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get candidate keypoints to be considered the root for a pose. Score for the
    keypoint must be >= all neighboring scores (i.e. equal to the max-pooled
    value) and above `score_threshold`.

    Parameters
    ----------
    score_threshold
        Minimum score for a keypoint to be considered as a root.
    scores
        See `decode_multiple_poses`.

    Returns
    -------
    scores_vec : np.ndarray
        Scores for each keypoint to be considered.
    max_loc_idx : np.ndarray
        Indices of the considered keypoints. Shape (N, 3) where the 3 indices
        map to the dimensions of the scores tensor with shape (17, h, w).
    """
    # The local maxima are recomputed from `scores` rather than read from a
    # separate model output: on device the two would be quantized
    # independently, so the `==` comparison would rarely hold -> zero peaks ->
    # no keypoints detected. Recomputing keeps the equality exact.
    max_pooled = max_pool_2d(scores, C.LOCAL_MAXIMUM_RADIUS)
    max_loc = (scores == max_pooled) & (scores >= score_threshold)
    max_loc_idx = np.argwhere(max_loc)
    scores_vec = scores[max_loc]
    # Stable descending sort, so keypoints with equal scores stay in raster
    # order. Quantized scores take only 256 distinct values, so ties are
    # common, and the order decides which poses NMS keeps.
    sort_idx = np.argsort(-scores_vec, kind="stable")
    return scores_vec[sort_idx], max_loc_idx[sort_idx]


def decode_multiple_poses(
    scores: np.ndarray,
    offsets: np.ndarray,
    displacements_fwd: np.ndarray,
    displacements_bwd: np.ndarray,
    max_pose_detections: int = C.MAX_POSE_DETECTIONS,
    score_threshold: float = C.SCORE_THRESHOLD,
    nms_radius: int = C.NMS_RADIUS,
    min_pose_score: float = C.MIN_POSE_SCORE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw model outputs into keypoint coordinates. Can detect multiple
    poses in the same image, up to `max_pose_detections`. This model has 17
    candidate keypoints it predicts. In this docstring, (h, w) correspond to
    height and width of the grid and are roughly equal to input image size
    divided by `OUTPUT_STRIDE`.

    Parameters
    ----------
    scores
        Array of scores in range [0, 1] indicating probability a candidate pose
        is real. Shape [17, h, w].
    offsets
        Array of offsets for a given keypoint, relative to the grid point.
        Shape [34, h, w].
    displacements_fwd
        When tracing the points for a pose, given a source keypoint, this value
        gives the displacement to the next keypoint in the pose. There are 16
        connections from one keypoint to another (it's a minimum spanning tree).
        Shape [32, h, w].
    displacements_bwd
        Same as displacements_fwd, except when traversing keypoint connections
        in the opposite direction.
    max_pose_detections
        Maximum number of distinct poses to detect in a single image.
    score_threshold
        Minimum score for a keypoint to be considered the root for a pose.
    nms_radius
        If two candidate keypoints for the same body part are within this distance,
        they are considered the same, and the lower confidence one discarded.
    min_pose_score
        Minimum confidence that a pose exists for it to be displayed.

    Returns
    -------
    pose_scores : np.ndarray
        Numpy array of pose confidence scores.
    pose_keypoint_scores : np.ndarray
        Numpy array of keypoint confidence scores.
    pose_keypoint_coords : np.ndarray
        Numpy array of keypoint coordinates in (y, x) format.
    """
    part_scores, part_idx = build_part_with_score(score_threshold, scores)

    height = scores.shape[1]
    width = scores.shape[2]
    # change dimensions from (x, h, w) to (x//2, h, w, 2) to allow return of complete coord array
    offsets = offsets.reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    displacements_fwd = displacements_fwd.reshape(2, -1, height, width).transpose(
        (1, 2, 3, 0)
    )
    displacements_bwd = displacements_bwd.reshape(2, -1, height, width).transpose(
        (1, 2, 3, 0)
    )

    pose_count = 0
    pose_scores = np.zeros(max_pose_detections)
    pose_keypoint_scores = np.zeros((max_pose_detections, C.NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((max_pose_detections, C.NUM_KEYPOINTS, 2))

    for root_score, (root_id, root_coord_y, root_coord_x) in zip(
        part_scores, part_idx, strict=False
    ):
        root_coord = np.array([root_coord_y, root_coord_x])
        root_image_coords = (
            root_coord * C.OUTPUT_STRIDE + offsets[root_id, root_coord_y, root_coord_x]
        )

        if within_nms_radius_fast(
            pose_keypoint_coords[:pose_count, root_id, :],
            nms_radius,
            root_image_coords,
        ):
            continue

        keypoint_scores, keypoint_coords = decode_pose(
            root_score,
            root_id,
            root_image_coords,
            scores,
            offsets,
            displacements_fwd,
            displacements_bwd=displacements_bwd,
        )

        pose_score = get_instance_score_fast(
            pose_keypoint_coords[:pose_count, :, :],
            nms_radius,
            keypoint_scores,
            keypoint_coords,
        )

        # Set min_pose_score to 0.0 to accept every decoded pose.
        if pose_score >= min_pose_score:
            pose_scores[pose_count] = pose_score
            pose_keypoint_scores[pose_count, :] = keypoint_scores
            pose_keypoint_coords[pose_count, :, :] = keypoint_coords
            pose_count += 1

        if pose_count >= max_pose_detections:
            break

    return pose_scores, pose_keypoint_scores, pose_keypoint_coords
