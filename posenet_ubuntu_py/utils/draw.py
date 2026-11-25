# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from qai_hub_apps_utils.draw import draw_connections, draw_points

import utils.constants as C


def get_adjacent_keypoints(
    keypoint_scores: np.ndarray, score_threshold: float
) -> list[tuple[int, int]]:
    """
    Compute which keypoint pairs should be connected by a skeleton edge.

    Parameters
    ----------
    keypoint_scores
        Scores for all candidate keypoints in the pose.
    score_threshold
        If either keypoint in a candidate edge is below this threshold, omit the edge.

    Returns
    -------
    results : list[tuple[int, int]]
        List of (src index, dst index) keypoint pairs to connect.
    """
    return [
        (left, right)
        for left, right in C.CONNECTED_PART_INDICES
        if keypoint_scores[left] >= score_threshold
        and keypoint_scores[right] >= score_threshold
    ]


def draw_skel_and_kp(
    img: np.ndarray,
    instance_scores: np.ndarray,
    keypoint_scores: np.ndarray,
    keypoint_coords: np.ndarray,
    min_pose_score: float = C.MIN_POSE_SCORE,
    min_part_score: float = C.MIN_PART_SCORE,
) -> None:
    """
    Draw the keypoints and edges on the input numpy array image in-place.

    Parameters
    ----------
    img
        Numpy array of the image, shape [H, W, 3].
    instance_scores
        Numpy array of confidence for each pose.
    keypoint_scores
        Numpy array of confidence for each keypoint.
    keypoint_coords
        Numpy array of coordinates for each keypoint, in (y, x) format.
    min_pose_score
        Minimum score for a pose to be displayed.
    min_part_score
        Minimum score for a keypoint to be displayed.
    """
    points = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        # Convert this pose's (y, x) keypoints to (x, y) for drawing.
        pose_points = keypoint_coords[ii, :, ::-1]
        connections = get_adjacent_keypoints(keypoint_scores[ii, :], min_part_score)
        if connections:
            draw_connections(img, pose_points, connections, (0, 255, 0), 2)

        for ks, kc in zip(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], strict=False
        ):
            if ks < min_part_score:
                continue
            points.append([kc[1], kc[0]])

    if points:
        draw_points(img, np.array(points), color=(0, 0, 255))
