# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import math
from typing import Tuple

import cv2
import numpy as np


def denormalize_coordinates(
    coordinates: np.ndarray,
    input_img_size: Tuple[int, int],
    scale: float = 1.0,
    pad: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    Maps detection coordinates from normalized [0, 1] to absolute coordinates in
    the original (pre-resize) image.

    Parameters
    ----------
    coordinates : np.ndarray
        Tensor of shape [..., 2]. Coordinates are ordered (y, x) and normalized to [0,1].
        This array is modified in place.

    input_img_size : (int, int)
        (H, W) of the network input (the resized padded tensor).

    scale : float
        Scale factor used during resizing to network size.

    pad : (int, int)
        Padding (H_pad, W_pad) added during resize-to-network.

    Returns
    -------
    np.ndarray
        Denormalized coordinates.
    """
    img_0, img_1 = input_img_size
    pad_0, pad_1 = pad
    denorm_coordinates = coordinates.copy()

    # Convert normalized coordinates -> network pixel space -> remove padding -> unscale -> int
    denorm_coordinates[..., 0] = (
        (denorm_coordinates[..., 0] * img_0 - pad_0) / scale
    ).astype(np.int32)
    denorm_coordinates[..., 1] = (
        (denorm_coordinates[..., 1] * img_1 - pad_1) / scale
    ).astype(np.int32)

    return denorm_coordinates


def apply_batched_affines_to_frame(
    frame: np.ndarray, affines: list[np.ndarray], output_image_size: tuple[int, int]
) -> np.ndarray:
    """
    Generate one image per affine applied to the given frame.
    I/O is numpy since this uses cv2 APIs under the hood.

    Parameters
    ----------
    frame: np.ndarray
        Frame on which to apply the affine. Shape is [H, W, C], dtype must be np.byte.

    affines: List[np.ndarray]
        List of 2x3 affine matrices to apply to the frame.

    output_image_size: Tuple[int, int]
        Size of each output frame.

    Returns
    -------
    np.ndarray
        Computed images. Shape is [B, H, W, C]
    """
    assert (
        frame.dtype == np.byte
        or frame.dtype
        == np.uint8  # noqa: PLR1714 Using a set for comparison is not equivalent to using == on both of these individually.
    )  # cv2 does not work correctly otherwise. Don't remove this assertion.

    imgs = []
    for affine in affines:
        img = cv2.warpAffine(frame, affine, output_image_size)
        imgs.append(img)
    return np.stack(imgs)


def apply_affine_to_coordinates(
    coordinates: np.ndarray, affine: np.ndarray
) -> np.ndarray:
    """
    Apply the given affine matrix to the given coordinates.

    Parameters
    ----------
    coordinates: np.ndarray
        Coordinates on which to apply the affine. Shape is [ ..., 2 ], where 2 == [X, Y]

    affines: np.ndarray
        Affine matrix to apply to the coordinates.

    Returns
    -------
        Transformed coordinates. Shape is [ ..., 2 ], where 2 == [X, Y]
    """
    return (affine[:, :2] @ coordinates.T + affine[:, 2:]).T


def compute_vector_rotation(
    vec_start: np.ndarray,
    vec_end: np.ndarray,
    offset_rads: float | np.ndarray = 0,
) -> np.ndarray:
    """
    From the given vector, compute the rotation angle of the vector with an added offset.

    Parameters
    ----------
    vec_start : np.ndarray
        Starting point of the vector. Shape [B, 2] (x, y).

    vec_end : np.ndarray
        Ending point of the vector. Shape [B, 2] (x, y).

    offset_rads : float or np.ndarray
        Offset (in radians) to subtract from the computed rotation.
        Can be a scalar or array broadcastable to shape [B].

    Returns
    -------
    theta : np.ndarray
        Rotation angle in radians. Shape [B].
    """
    # Compute dy, dx
    dy = vec_start[..., 1] - vec_end[..., 1]
    dx = vec_start[..., 0] - vec_end[..., 0]

    # atan2(dy, dx)
    theta = np.arctan2(dy, dx) - offset_rads
    return theta


def resize_pad(
    image: np.ndarray,
    dst_size: Tuple[int, int],
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize and pad image to shape (dst_size[0], dst_size[1]) while preserving aspect ratio.

    Parameters
    ----------
    image
        Input image with shape [H, W] or [H, W, C]. dtype can be uint8, float32, etc.

    dst_size
        Desired (height, width).

    Returns
    -------
    rescaled_padded_image : np.ndarray
        Output image with shape (dst_h, dst_w) or (dst_h, dst_w, C).

    scale : float
        Scale factor applied to the original image (same for H and W).

    padding : (int, int)
        (pad_left, pad_top) applied to the resized image.
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")

    src_h, src_w = image.shape[:2]
    dst_h, dst_w = int(dst_size[0]), int(dst_size[1])

    # Compute uniform scale to fit within dst while preserving aspect ratio
    h_ratio = dst_h / src_h
    w_ratio = dst_w / src_w
    scale = min(h_ratio, w_ratio)

    new_h = max(1, math.floor(src_h * scale))
    new_w = max(1, math.floor(src_w * scale))

    interp = cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Compute padding amounts
    pad_total_h = dst_h - new_h
    pad_total_w = dst_w - new_w

    pad_top, pad_bottom = (pad_total_h // 2, pad_total_h - pad_total_h // 2)
    pad_left, pad_right = (pad_total_w // 2, pad_total_w - pad_total_w // 2)

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0.0,
    )

    return padded, scale, (pad_left, pad_top)
