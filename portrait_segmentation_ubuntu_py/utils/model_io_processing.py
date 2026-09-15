# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np

import utils.constants as C


def decode_mask(model_output: np.ndarray) -> np.ndarray:
    """
    Convert the model's per-pixel 2-class logits into a foreground probability map.

    A softmax over the two classes reduces to a sigmoid of their difference,
    giving a per-pixel foreground probability that downstream post-processing can
    threshold and smooth.

    Parameters
    ----------
    model_output
        Model output tensor of float logits, batch-first and 4-D. The class axis
        (2 classes) may be first or last (NCHW or NHWC).

    Returns
    -------
    np.ndarray
        Foreground probability map of shape (H, W), dtype float32, in [0, 1].
    """
    logits = model_output[0]
    class_axis = int(np.argmin(logits.shape))
    if class_axis != logits.ndim - 1:
        logits = np.moveaxis(logits, class_axis, -1)
    diff = (
        logits[..., C.FOREGROUND_CHANNEL] - logits[..., 1 - C.FOREGROUND_CHANNEL]
    ).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-np.clip(diff, -30.0, 30.0)))
    return prob.astype(np.float32)


def _clean_mask(prob: np.ndarray) -> np.ndarray:
    """
    Threshold at ``C.MASK_THRESHOLD`` and morphologically open then close.

    Parameters
    ----------
    prob
        Foreground probability map of shape (h, w), dtype float32, in [0, 1].

    Returns
    -------
    np.ndarray
        Binary mask of shape (h, w), dtype uint8 (values 0 or 1).
    """
    mask = (prob >= C.MASK_THRESHOLD).astype(np.uint8)

    if C.MORPH_KERNEL_SIZE > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (C.MORPH_KERNEL_SIZE, C.MORPH_KERNEL_SIZE)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def postprocess_mask(
    prob: np.ndarray,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """
    Clean a probability map and upsample it to a frame-resolution binary mask.

    Parameters
    ----------
    prob
        Foreground probability map of shape (h, w), dtype float32, in [0, 1], at
        the model's output resolution.
    frame_size
        Target ``(width, height)`` of the frame the mask is applied to.

    Returns
    -------
    np.ndarray
        Binary foreground mask of shape (height, width), dtype bool.
    """
    frame_w, frame_h = frame_size
    cleaned = _clean_mask(prob).astype(np.float32)
    upsampled = cv2.resize(cleaned, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    return upsampled >= 0.5


def build_soft_alpha(
    prob: np.ndarray,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """
    Build a soft, frame-resolution foreground alpha for background compositing.

    The probability is cleaned with ``_clean_mask``, feathered with a Gaussian
    blur for a soft edge, then upsampled to the frame resolution.

    Parameters
    ----------
    prob
        Foreground probability map of shape (h, w), dtype float32, in [0, 1], at
        the model's output resolution.
    frame_size
        Target ``(width, height)`` of the frame the alpha is applied to.

    Returns
    -------
    np.ndarray
        Foreground alpha of shape (height, width), dtype float32, in [0, 1].
    """
    frame_w, frame_h = frame_size
    alpha = _clean_mask(prob).astype(np.float32)
    if C.EDGE_FEATHER > 0:
        alpha = cv2.GaussianBlur(alpha, (C.EDGE_FEATHER, C.EDGE_FEATHER), 0)
    return cv2.resize(alpha, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


def blur_background(rgb_frame: np.ndarray) -> np.ndarray:
    """
    Return a heavily blurred copy of the frame for use as a background.

    Downscaling then upscaling by ``C.BLUR_DOWNSCALE`` yields a strong blur far
    more cheaply than a large-kernel Gaussian at full resolution.

    Parameters
    ----------
    rgb_frame
        RGB frame of shape [H, W, 3], dtype uint8.

    Returns
    -------
    np.ndarray
        Blurred RGB frame of the same shape and dtype.
    """
    frame_h, frame_w = rgb_frame.shape[:2]
    small_w = max(1, frame_w // C.BLUR_DOWNSCALE)
    small_h = max(1, frame_h // C.BLUR_DOWNSCALE)
    small = cv2.resize(rgb_frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


def resize_to_fit(image: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """
    Fit an image to the frame size: center-crop if it is larger, stretch if smaller.

    Parameters
    ----------
    image
        RGB image of shape [H, W, 3], dtype uint8.
    frame_size
        Target ``(width, height)``.

    Returns
    -------
    np.ndarray
        RGB image of shape [height, width, 3], dtype uint8.
    """
    frame_w, frame_h = frame_size
    img_h, img_w = image.shape[:2]
    if img_w >= frame_w and img_h >= frame_h:
        x0 = (img_w - frame_w) // 2
        y0 = (img_h - frame_h) // 2
        return image[y0 : y0 + frame_h, x0 : x0 + frame_w]
    return cv2.resize(image, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


def composite_background(
    rgb_frame: np.ndarray,
    alpha: np.ndarray,
    background: np.ndarray,
) -> None:
    """
    Composite the foreground over a background using a soft alpha, in place.

    Parameters
    ----------
    rgb_frame
        Original RGB frame of shape [H, W, 3], dtype uint8. Modified in place.
    alpha
        Foreground alpha of shape (H, W), dtype float32, in [0, 1].
    background
        Background RGB image of shape [H, W, 3], dtype uint8, same size as the
        frame.
    """
    a = alpha[..., None]
    blended = rgb_frame.astype(np.float32) * a + background.astype(np.float32) * (
        1.0 - a
    )
    rgb_frame[:] = blended.astype(np.uint8)


def blend_mask(
    rgb_frame: np.ndarray,
    mask: np.ndarray,
) -> None:
    """
    Blend the foreground mask onto the original frame in place.

    Foreground pixels are tinted with ``C.OVERLAY_COLOR`` at ``C.OVERLAY_ALPHA``.

    Parameters
    ----------
    rgb_frame
        Original RGB frame of shape [H, W, 3], dtype uint8. Modified in place.
    mask
        Binary foreground mask of shape (H, W), dtype bool (True == foreground),
        at the same resolution as ``rgb_frame``.
    """
    overlay = np.array(C.OVERLAY_COLOR, dtype=np.float32)
    alpha = C.OVERLAY_ALPHA
    rgb_frame[mask] = (
        rgb_frame[mask].astype(np.float32) * (1.0 - alpha) + overlay * alpha
    ).astype(np.uint8)
