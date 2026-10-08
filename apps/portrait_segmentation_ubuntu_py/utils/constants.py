# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Directory (relative to the app root) holding the model file and its
# metadata.json
MODELS_DIR = "models"

# The model emits a per-pixel, 2-class segmentation map. Channel 1 is the foreground (person) logit and channel 0
# is the background logit;
FOREGROUND_CHANNEL = 1

# Minimum foreground probability (softmax over the 2 classes, in [0, 1]) for a
# pixel to count as foreground. Above the 0.5 argmax point so low-confidence
# pixels near the person's edge don't flicker in/out as spurious detections.
MASK_THRESHOLD = 0.75

# Side length (px) of the elliptical kernel used for the morphological open/close
# cleanup. Open removes isolated speckles; close fills small holes inside the
# person. 0 disables the morphological step.
MORPH_KERNEL_SIZE = 3

# RGB color the foreground (person) mask is blended with for visualization.
OVERLAY_COLOR = (68, 132, 255)

# Blend strength of the overlay over the foreground pixels, in [0, 1].
OVERLAY_ALPHA = 0.5

# Side length of the Gaussian kernel used to feather the mask edge into a soft
# alpha for background blur/replacement, as a fraction of the frame's shorter
# side. 0 disables feathering.
EDGE_FEATHER_FRACTION = 0.008

# Downscale factor for the background blur.
BLUR_DOWNSCALE = 4

# Side length (px) of the blur kernel applied at the downscaled size. 0 disables
# it, which makes the background visibly blocky.
# Side length (px) of the blur kernel applied at the downscaled size. Must be odd
# (or 0 to disable, which makes the background visibly blocky).
BLUR_KERNEL_SIZE = 5
