# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
from PIL import Image

import utils.constants as C


def preprocess_image(image_path: str) -> tuple[Image.Image, int, int, np.ndarray]:
    """Load an image, squash-resize to the backbone input size, and return NCHW float32.

    Parameters
    ----------
    image_path
        Path to the input image file.

    Returns
    -------
    tuple[Image.Image, int, int, np.ndarray]
        Original PIL image, original width, original height, and an
        [1, 3, H, W] float32 array normalized to [0, 1].
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    arr = (
        np.asarray(
            img.resize((C.BACKBONE_SIZE, C.BACKBONE_SIZE), Image.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    nchw = np.transpose(arr, (2, 0, 1))[np.newaxis].copy()
    return img, orig_w, orig_h, nchw
