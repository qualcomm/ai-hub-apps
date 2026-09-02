# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import utils.constants as C


def draw_predictions(
    image: Image.Image,
    boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
) -> Image.Image:
    """Render mask overlay and labeled boxes onto a copy of the input image.

    Parameters
    ----------
    image
        Original PIL RGB image.
    boxes
        [N, 4] absolute pixel boxes (x1, y1, x2, y2).
    masks
        [N, H, W] float mask logits produced by the head.
    scores
        [N] confidence scores.
    labels
        Length-N list of prompt labels.

    Returns
    -------
    Image.Image
        Image with translucent colored masks and box labels drawn on top.
    """
    orig_w, orig_h = image.size
    canvas = np.asarray(image, dtype=np.float32).copy()
    for idx in range(len(scores)):
        color = np.array(
            C.OVERLAY_COLORS[idx % len(C.OVERLAY_COLORS)], dtype=np.float32
        )
        mask_up = np.asarray(
            Image.fromarray(masks[idx].astype(np.float32)).resize(
                (orig_w, orig_h), Image.BILINEAR
            )
        )
        canvas[mask_up > C.MASK_THRESHOLD] = (
            canvas[mask_up > C.MASK_THRESHOLD] * 0.5 + color * 0.5
        )
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)

    # Scale line width and font size to the image so boxes/labels stay legible
    # on both small crops and full-resolution phone photos (e.g. 4032x3024).
    scale = min(orig_w, orig_h) / 1000
    halo_width = max(2, round(4 * scale))
    box_width = max(1, round(2 * scale))
    font_size = max(12, round(28 * scale))
    font = ImageFont.load_default(size=font_size)

    for idx in range(len(scores)):
        color = C.OVERLAY_COLORS[idx % len(C.OVERLAY_COLORS)]
        x1, y1, x2, y2 = (float(v) for v in boxes[idx])
        # The mask overlay is drawn in the same color, so a thin same-color box
        # washes out against it. Draw a dark halo underneath, then the colored
        # box on top, so the outline reads clearly regardless of the mask.
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=halo_width)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        label = f"{labels[idx]}: {scores[idx]:.2f}"
        tx, ty = x1 + 4 * scale, y1 + 2 * scale
        # Filled background behind the label so text stays legible over any mask.
        left, top, right, bottom = draw.textbbox((tx, ty), label, font=font)
        pad = max(1, round(2 * scale))
        draw.rectangle(
            [left - pad, top - pad, right + pad, bottom + pad], fill=(0, 0, 0)
        )
        draw.text((tx, ty), label, fill=color, font=font)
    return out
