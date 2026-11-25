# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np


def preprocess(
    rgb_frame: np.ndarray, model_input_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare a colourization model's inputs from a single RGB frame.

    DDColor (and colourization models generally) work in CIE Lab space: the model
    is fed the lightness channel and predicts the two chroma channels. The frame
    is therefore reduced to its L channel, expanded back into a 3-channel
    "grayscale RGB" image at the model's input resolution, and normalized to
    [0, 1]. The full-resolution L channel is returned separately so the predicted
    chroma can be recombined with the original detail during postprocessing.

    This mirrors ``DDColorApp.colorize`` in AI Hub Models so the app reproduces
    the reference demo.

    Parameters
    ----------
    rgb_frame
        Input RGB image of shape [H, W, 3], dtype uint8 in range [0, 255].
    model_input_shape
        Model input (height, width) to resize the grayscale image to.

    Returns
    -------
    gray_rgb : np.ndarray
        Grayscale-as-RGB model input of shape [model_h, model_w, 3], dtype
        float32 in range [0, 1].
    orig_l : np.ndarray
        Lightness channel of the original frame, shape [H, W, 1], dtype float32
        (OpenCV Lab scale, L in [0, 100]).
    """
    img = rgb_frame.astype(np.float32) / 255.0

    # Full-resolution lightness, kept to restore detail in postprocessing.
    orig_l = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, :1]

    # Grayscale RGB at the model's input resolution: take L, zero the chroma,
    # convert back to RGB so the image looks gray but is 3-channel.
    model_h, model_w = model_input_shape
    img_resized = cv2.resize(img, (model_w, model_h))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_RGB2Lab)[:, :, :1]
    gray_lab = np.concatenate(
        (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1
    )
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_Lab2RGB)

    return gray_rgb.astype(np.float32), orig_l


def decode_chroma(model_output: np.ndarray) -> np.ndarray:
    """
    Convert a raw model output tensor into a 2-channel (a, b) chroma map.

    Colourization models emit the two Lab chroma channels with a batch dimension,
    laid out as either NCHW ``[1, 2, H, W]`` or NHWC ``[1, H, W, 2]``. The layout
    is detected from the output tensor itself rather than assumed from the input:
    an exported model can use NHWC input but NCHW output (as the DDColor TFLite
    asset does), so the two cannot be derived from a single flag.

    Parameters
    ----------
    model_output
        Model output tensor of floats, batch-first and 4-D with a channel dim of
        2 (NCHW or NHWC).

    Returns
    -------
    np.ndarray
        Chroma map of shape (H, W, 2), dtype float32.
    """
    chroma = np.asarray(model_output, dtype=np.float32)[0]  # drop batch axis
    if chroma.ndim == 3 and chroma.shape[-1] == 2:
        return chroma  # NHWC: [H, W, 2]
    if chroma.ndim == 3 and chroma.shape[0] == 2:
        return np.transpose(chroma, (1, 2, 0))  # NCHW: [2, H, W] -> [H, W, 2]
    raise ValueError(
        "Expected a 2-channel (a, b) chroma output laid out as [1, 2, H, W] or "
        f"[1, H, W, 2]; got shape {np.asarray(model_output).shape}"
    )


def colorize(
    chroma: np.ndarray,
    orig_l: np.ndarray,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """
    Recombine predicted chroma with the original lightness into an RGB image.

    The predicted (a, b) map is upscaled to the frame resolution and stacked on
    top of the original full-resolution L channel, so the output keeps the
    detail of the input while taking its color from the model. The Lab image is
    then converted back to RGB, matching the AI Hub Models colourization demo.

    Parameters
    ----------
    chroma
        Predicted chroma map of shape (model_h, model_w, 2), dtype float32, at
        the model's output resolution.
    orig_l
        Lightness channel of the original frame, shape [H, W, 1], dtype float32.
    frame_size
        Original frame size as (width, height).

    Returns
    -------
    np.ndarray
        Colourized RGB image of shape (H, W, 3), dtype uint8.
    """
    width, height = frame_size
    chroma_resized = cv2.resize(chroma, (width, height), interpolation=cv2.INTER_LINEAR)
    output_lab = np.concatenate((orig_l, chroma_resized), axis=-1)
    output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_Lab2RGB)
    return np.clip(output_rgb * 255.0, 0, 255).round().astype(np.uint8)
