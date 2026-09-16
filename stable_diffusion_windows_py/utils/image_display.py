# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Image helpers: convert a float image to uint8 and display or save it."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Converts a numpy array image to uint8 type. Values are clipped to the
    range [0, 1] before scaling to [0, 255].
    """
    clipped_image = np.clip(image, 0, 1)
    return np.round(clipped_image * 255).astype(np.uint8)


def display_or_save_image(
    image: Image.Image,
    output_dir: str | None = None,
    filename: str = "image.png",
    desc: str = "image",
) -> None:
    """
    If output_dir is set, save image to disk.
    Else try to display image; if that fails, save to disk in a default location.
    """
    if output_dir is not None:
        _save_image(image, output_dir, filename, desc)
        return

    try:
        print(f"Displaying {desc}")
        image.show()
        return
    except Exception:
        print("Failed to display image; saving to disk instead.")

    _save_image(image, os.path.join(os.getcwd(), "build"), filename, desc)


def _save_image(image: Image.Image, base_dir: str, filename: str, desc: str) -> None:
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, filename)
    image.save(path)
    print(f"Saving {desc} to {path}")
