# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np
from qai_hub_apps_utils.image_processing import resize_pad

import utils.constants as C


def load_image_rgb(path: str) -> np.ndarray:
    """
    Load an image from disk as an RGB numpy array.

    Parameters
    ----------
    path
        Path to the image file.

    Returns
    -------
    np.ndarray
        RGB image of shape [H, W, 3], dtype uint8.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(rgb_image: np.ndarray) -> np.ndarray:
    """
    Preprocess an RGB image into the CavaFace model input tensor.

    The image is resized and zero-padded to the model resolution while
    preserving aspect ratio (matching the qai_hub_models reference
    preprocessing), then scaled to [0, 1]. The exported model bakes in the
    remaining mean/std normalization, so no further scaling is applied here.

    Parameters
    ----------
    rgb_image
        RGB image of shape [H, W, 3], dtype uint8.

    Returns
    -------
    np.ndarray
        Model input of shape [1, INPUT_HEIGHT, INPUT_WIDTH, 3], dtype float32,
        with values in [0, 1].
    """
    normalized = rgb_image.astype(np.float32) / 255.0
    resized, _, _ = resize_pad(normalized, (C.INPUT_HEIGHT, C.INPUT_WIDTH))
    return np.expand_dims(resized, axis=0)


def get_gstreamer_input_pipeline(
    video_source: str, video_source_size: tuple[int, int]
) -> str:
    """
    Build a GStreamer pipeline string for reading frames from a given video source,
    converting them to RGB, and exposing them via an appsink.

    Parameters
    ----------
    video_source
        The left-hand side of the pipeline specifying the source element and its
        properties, e.g. ``"v4l2src device=/dev/video0"`` or
        ``"filesrc location=video.mp4 ! decodebin"``. The trailing ``!`` is
        appended by this function.
    video_source_size
        The (width, height) of the incoming video frames expected from
        ``video_source``. Used to set the caps on both the NV12 and RGB segments.

    Returns
    -------
    str
        A GStreamer pipeline description string.

    Notes
    -----
    - Requires the ``qtivtransform`` element (part of Qualcomm/Hexagon/GStreamer
      plugins) if ``qtiqmmfsrc`` is used; ensure it is available in your
      GStreamer setup.
    """
    video_source_width, video_source_height = video_source_size

    if video_source.lower().strip().startswith("qtiqmmfsrc"):
        return (
            f"{video_source} ! "
            f"video/x-raw,width={video_source_width},height={video_source_height},framerate=60/1,format=NV12 ! "
            f"videoconvert ! video/x-raw,format=RGB,width={video_source_width},height={video_source_height} ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            "appsink name=appsink drop=true sync=false max-buffers=1 emit-signals=true"
        )

    return (
        f"{video_source} ! "
        "videoconvert ! videoscale ! "
        f"video/x-raw,width={video_source_width},height={video_source_height},format=RGB ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink name=appsink drop=true sync=false max-buffers=1 emit-signals=true"
    )
