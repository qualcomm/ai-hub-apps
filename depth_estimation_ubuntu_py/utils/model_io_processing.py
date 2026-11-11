# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np


def image_input_layout(input_shape: list[int]) -> tuple[bool, int, int]:
    """
    Read an image model's layout from its 4-D input shape.

    Parameters
    ----------
    input_shape
        Input tensor shape, NHWC or NCHW, with a channel dim of 3.

    Returns
    -------
    tuple[bool, int, int]
        ``(channels_first, height, width)``.

    Raises
    ------
    ValueError
        If the shape is not 4-D with a channel dim of 3.
    """
    if len(input_shape) != 4 or 3 not in input_shape[1:]:
        raise ValueError(
            "Expected a 4-D input shape with a channel dim of 3 (NHWC or "
            f"NCHW); got {input_shape}"
        )
    if input_shape.index(3, 1) == 1:
        return True, input_shape[2], input_shape[3]
    return False, input_shape[1], input_shape[2]


def decode_depth(model_output: np.ndarray) -> np.ndarray:
    """
    Convert a raw model output tensor into a 2-D depth map.

    Depth-estimation models emit a single-channel map with a batch dimension,
    laid out as either NCHW ``[1, 1, H, W]`` or NHWC ``[1, H, W, 1]``. Both
    reduce to ``(H, W)`` once the singleton batch and channel axes are removed.

    Parameters
    ----------
    model_output
        Model output tensor of floats, batch-first and 4-D with a channel dim
        of 1 (NCHW or NHWC).

    Returns
    -------
    np.ndarray
        Depth map of shape (H, W), dtype float32. Larger values are nearer the
        camera (relative, unitless depth).
    """
    depth = np.squeeze(np.asarray(model_output, dtype=np.float32))
    if depth.ndim != 2:
        raise ValueError(
            "Expected a single-channel depth output that squeezes to 2-D (H, W); "
            f"got shape {np.asarray(model_output).shape}"
        )
    return depth


def undo_resize_pad(
    depth: np.ndarray,
    frame_size: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
) -> np.ndarray:
    """
    Undo the resize-and-pad applied before inference, mapping a depth map from
    network space back to the original frame resolution.

    This mirrors ``qai_hub_models.utils.image_processing.undo_resize_pad`` so the
    app reproduces the reference AI Hub Models demo: the padded network-space map
    is upscaled by ``1 / scale`` and the padding is cropped away.

    Parameters
    ----------
    depth
        Depth map of shape (net_h, net_w), at the model's input resolution.
    frame_size
        Original frame size as (width, height).
    scale
        Scale factor that ``resize_pad`` applied to the original frame.
    pad
        (pad_left, pad_top) padding that ``resize_pad`` added, in network pixels.

    Returns
    -------
    np.ndarray
        Depth map of shape (height, width) matching the original frame.
    """
    width, height = frame_size
    pad_left, pad_top = pad

    rescaled = cv2.resize(
        depth, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR
    )

    crop_left = round(pad_left / scale)
    crop_top = round(pad_top / scale)
    return rescaled[crop_top : crop_top + height, crop_left : crop_left + width]


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """
    Turn a depth map into an RGB heatmap for visualization.

    The depth is normalized to [0, 1] by its own maximum and mapped through the
    "plasma" colormap, matching the AI Hub Models depth-estimation demo. OpenCV's
    ``COLORMAP_PLASMA`` is used (equivalent to matplotlib's "plasma" to within
    1/255), so the app needs no matplotlib dependency.

    Parameters
    ----------
    depth
        Depth map of shape (H, W), dtype float32.

    Returns
    -------
    np.ndarray
        RGB heatmap of shape (H, W, 3), dtype uint8.
    """
    depth_max = float(depth.max())
    normalized = depth / depth_max if depth_max > 0 else np.zeros_like(depth)
    indices = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(indices, cv2.COLORMAP_PLASMA)
    return bgr[..., ::-1]
