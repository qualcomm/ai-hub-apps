# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

import utils.constants as C


def preprocess_frame(rgb_frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Preprocess one RGB frame into the model's spatial input.

    Resizes preserving aspect ratio so the shorter side is the crop size
    divided by ``C.CROP_RATIO``, center-crops to (target_h, target_w), and
    scales to [0, 1].

    Parameters
    ----------
    rgb_frame
        Input RGB frame of shape [H, W, 3], dtype uint8 in range [0, 255].
    target_h
        Target height the model expects.
    target_w
        Target width the model expects.

    Returns
    -------
    np.ndarray
        Preprocessed frame of shape [target_h, target_w, 3], dtype float32
        in [0, 1].
    """
    h, w = rgb_frame.shape[:2]
    resize_h, resize_w = (
        round(target_h / C.CROP_RATIO),
        round(target_w / C.CROP_RATIO),
    )
    scale = max(resize_h / h, resize_w / w)
    resized = _resize(rgb_frame, round(w * scale), round(h * scale))
    return _center_crop(resized, target_h, target_w).astype(np.float32) / 255.0


def preprocess_frame_kinetics(
    rgb_frame: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Preprocess one RGB frame for a 112x112 Kinetics-400 model.

    Resizes to a fixed 128x171 regardless of aspect ratio, then center-crops.

    Parameters
    ----------
    rgb_frame
        Input RGB frame of shape [H, W, 3], dtype uint8 in range [0, 255].
    target_h
        Target height the model expects.
    target_w
        Target width the model expects.

    Returns
    -------
    np.ndarray
        Preprocessed frame of shape [target_h, target_w, 3], dtype float32
        in [0, 1].
    """
    resized = cv2.resize(
        rgb_frame,
        (C.KINETICS_RESIZE_W, C.KINETICS_RESIZE_H),
        interpolation=cv2.INTER_LINEAR,
    )
    return _center_crop(resized, target_h, target_w).astype(np.float32) / 255.0


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    # INTER_AREA antialiases when shrinking, which keeps downscaled frames
    # much closer to how the training pipeline produced them. The Kinetics
    # recipe above instead needs plain bilinear, without antialiasing.
    shrinking = width < frame.shape[1] or height < frame.shape[0]
    return cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )


def _center_crop(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    top = round((frame.shape[0] - target_h) / 2.0)
    left = round((frame.shape[1] - target_w) / 2.0)
    return frame[top : top + target_h, left : left + target_w]


def clip_to_input(
    frames: Sequence[np.ndarray], channels_first: bool = True
) -> np.ndarray:
    """Stack the clip's preprocessed frames into the model's input tensor.

    Parameters
    ----------
    frames
        Preprocessed frames, each of shape [H, W, 3], dtype float32 in [0, 1].
    channels_first
        True to emit ``[1, C, T, H, W]``, False to emit ``[1, T, H, W, C]``.

    Returns
    -------
    np.ndarray
        Clip tensor with a leading batch axis, dtype float32.
    """
    clip = np.stack(list(frames), axis=0)
    if channels_first:
        clip = np.transpose(clip, (3, 0, 1, 2))
    return clip[np.newaxis, ...].astype(np.float32)
