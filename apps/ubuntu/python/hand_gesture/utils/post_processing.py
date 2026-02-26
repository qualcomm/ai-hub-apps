# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Tuple

import numpy as np


def decode_preds_from_anchors(
    box_coords: np.ndarray, img_size: Tuple[int, int], anchors: np.ndarray
) -> None:
    """
    Decode predictions using the provided anchors.

    Notes
    -----
    - Operates in-place on `box_coords`.
    - Supports additional coordinates (e.g., keypoints) beyond the first 2 rows:
        - row 0: (x_center, y_center)
        - row 1: (w, h)
        - rows 2+: interpreted as keypoints ... each entry is (x, y)
    - Ranges expected to be normalized (0..1) before scaling by anchors and image size.

    Parameters
    ----------
    box_coords : np.ndarray
        Shape [..., K, 2] where the last dim is (x, y) and K >= 2.
        Convention for the first two rows:
            box_coords[..., 0, :] = (x_center, y_center)
            box_coords[..., 1, :] = (w, h)
        Any rows from index 2 onward are treated as keypoints and decoded similarly.
        This array is updated in place.

    img_size : (int, int)
        (width, height) of the network input (NOT original image size).
        The order matches how coordinates are treated.

    anchors : np.ndarray
        Same leading shape as `box_coords` except the middle K must match for at least
        the first two rows; typically the same K. Expected shape [..., K, 2] where:
            anchors[..., 0, :] = (x_center, y_center)
            anchors[..., 1, :] = (w, h)
        Values are in the same normalized space used by `box_coords` before decoding.
    """
    assert (
        box_coords.shape[-1] == anchors.shape[-1] == 2
    ), f"Last dim must be 2 for (x, y); got {box_coords.shape[-1]} and {anchors.shape[-1]}"
    # The coord axis is the second-to-last; must match for at least first two rows
    assert (
        box_coords.shape[-2] >= 2 and anchors.shape[-2] >= 2
    ), "Need at least 2 rows (center and size) in both box_coords and anchors."

    w_size, h_size = img_size

    # Unpack anchor parts
    anchors_x = anchors[..., 0, 0]
    anchors_y = anchors[..., 0, 1]
    anchors_w = anchors[..., 1, 0]
    anchors_h = anchors[..., 1, 1]

    # Helper to add a trailing axis for broadcasting with keypoints rows [..., K-2, 2]
    anchors_w_exp = anchors_w[..., np.newaxis]  # shape [..., 1]
    anchors_h_exp = anchors_h[..., np.newaxis]  # shape [..., 1]
    anchors_x_exp = anchors_x[..., np.newaxis]  # shape [..., 1]
    anchors_y_exp = anchors_y[..., np.newaxis]  # shape [..., 1]

    # Decode center (x, y) using anchors and image size
    # x_center
    box_coords[..., 0, 0] = (box_coords[..., 0, 0] / w_size) * anchors_w + anchors_x
    # y_center
    box_coords[..., 0, 1] = (box_coords[..., 0, 1] / h_size) * anchors_h + anchors_y

    # Decode width/height in absolute (anchor-scaled) terms
    box_coords[..., 1, 0] = (box_coords[..., 1, 0] / w_size) * anchors_w
    box_coords[..., 1, 1] = (box_coords[..., 1, 1] / h_size) * anchors_h

    # If there are additional coordinates (e.g., keypoints), decode them similarly
    if box_coords.shape[-2] > 2:
        # For keypoints, we scale x and y using the anchor w/h and then add anchor center.
        # Shapes:
        #   box_coords[..., 2:, 0] -> [..., K-2]
        #   anchors_*_exp -> [..., 1]
        # Broadcasting will expand the trailing dimension to match K-2.
        box_coords[..., 2:, 0] = (
            box_coords[..., 2:, 0] / w_size
        ) * anchors_w_exp + anchors_x_exp
        box_coords[..., 2:, 1] = (
            box_coords[..., 2:, 1] / h_size
        ) * anchors_h_exp + anchors_y_exp
