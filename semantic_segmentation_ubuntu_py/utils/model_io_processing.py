# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np

import utils.constants as C


def _voc_colormap(num_classes: int) -> np.ndarray:
    """Build the standard Pascal VOC segmentation color palette.

    This is the canonical bit-interleaved VOC palette (class 0 == background == black),
    matching the colors used by the DeepLab reference implementations.

    Parameters
    ----------
    num_classes
        Number of classes to generate colors for.

    Returns
    -------
    np.ndarray
        Array of shape (num_classes, 3), dtype uint8, RGB colors per class.
    """
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        colormap[i] = (r, g, b)
    return colormap


# Precomputed once; indexed by predicted class id.
VOC_COLORMAP = _voc_colormap(C.NUM_CLASSES)

# Same palette as a 256-entry LUT so the color lookup runs via cv2.LUT (SIMD)
# instead of a numpy gather; rows past the class count stay black (unused).
_VOC_LUT = np.zeros((1, 256, 3), dtype=np.uint8)
_VOC_LUT[0, : VOC_COLORMAP.shape[0]] = VOC_COLORMAP


def decode_mask(model_output: np.ndarray) -> np.ndarray:
    """
    Convert the model output into a per-pixel class-id map.

    The VOC segmentation models (AI Hub w8a8 TFLite) perform the argmax
    on-device and emit the class-id mask directly: output ``mask`` has shape
    ``[1, H, W]``, dtype uint8 (per metadata.json). In that case this returns
    the mask as-is.

    As a fallback, a raw per-class-logits output is also supported: if a class
    axis of size ``C.NUM_CLASSES`` is present (NHWC ``[1, H, W, num_classes]``
    or NCHW ``[1, num_classes, H, W]``), the argmax over that axis is taken.

    Parameters
    ----------
    model_output
        Either a pre-argmaxed mask of shape [1, H, W], or per-class
        logits/probabilities of shape [1, H, W, num_classes] or
        [1, num_classes, H, W].

    Returns
    -------
    np.ndarray
        Class-id map of shape (H, W), dtype uint8, where each value is the
        predicted class index in [0, num_classes).
    """
    output = model_output[0]  # drop batch

    # Pre-argmaxed mask: already a 2-D class-id map (the VOC model case).
    if output.ndim == 2:
        return output.astype(np.uint8)

    # Fallback: raw per-class logits -> argmax over the class axis. The class
    # axis can only be the last (NHWC [H, W, num_classes]) or the first (NCHW
    # [num_classes, H, W]) axis of this 3-D tensor; a middle axis is always
    # spatial. Checking only those two positions avoids mis-selecting a spatial
    # dimension that happens to equal C.NUM_CLASSES. NHWC is preferred when both
    # ends match (the AI Hub channels-last convention).
    if output.ndim == 3 and output.shape[-1] == C.NUM_CLASSES:
        class_axis = output.ndim - 1  # NHWC
    elif output.ndim == 3 and output.shape[0] == C.NUM_CLASSES:
        class_axis = 0  # NCHW
    else:
        raise ValueError(
            f"Unexpected model output shape {model_output.shape}: not a 2-D mask, "
            f"and neither the first nor last axis equals {C.NUM_CLASSES} classes. "
            "Check the model / NUM_CLASSES."
        )

    class_map = np.argmax(output, axis=class_axis)
    return class_map.astype(np.uint8)


def blend_mask(
    rgb_frame: np.ndarray,
    class_map: np.ndarray,
) -> None:
    """
    Blend the per-class color overlay onto the original frame in place.

    The class map is colored with the Pascal VOC palette, and that color overlay
    is alpha-blended over the foreground pixels. Background pixels
    (``C.BACKGROUND_CLASS``) are left untouched.

    Every step is a SIMD OpenCV op on uint8 (no per-pixel numpy float work): the
    color lookup is a ``cv2.LUT`` at the model's (smaller) output resolution, only
    the resulting RGB overlay is upscaled, the blend is a ``cv2.addWeighted``, and
    the foreground-only write is a ``cv2.copyTo`` with a single-channel mask. This
    keeps the whole overlay off the per-frame critical path.

    Parameters
    ----------
    rgb_frame
        Original RGB frame of shape [H, W, 3], dtype uint8. Modified in place.
    class_map
        Per-pixel class-id map of shape (h, w), dtype uint8, at the model's
        output resolution.
    """
    frame_h, frame_w = rgb_frame.shape[:2]

    # Color at the model's output resolution (far fewer pixels than the frame) via
    # a LUT, then upscale the RGB overlay once. Nearest-neighbor keeps class
    # boundaries crisp; the contour follows the model grid.
    color_small = cv2.LUT(cv2.cvtColor(class_map, cv2.COLOR_GRAY2RGB), _VOC_LUT)
    color_mask = cv2.resize(
        color_small, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
    )
    # Single-channel foreground mask (nonzero == blend here) for cv2.copyTo.
    foreground = cv2.resize(
        (class_map != C.BACKGROUND_CLASS).astype(np.uint8),
        (frame_w, frame_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # SIMD uint8 alpha blend over the whole frame, then keep it only where the
    # mask is foreground (background pixels retain the original frame).
    alpha = C.OVERLAY_ALPHA
    blended = cv2.addWeighted(rgb_frame, 1.0 - alpha, color_mask, alpha, 0.0)
    cv2.copyTo(blended, foreground, rgb_frame)
