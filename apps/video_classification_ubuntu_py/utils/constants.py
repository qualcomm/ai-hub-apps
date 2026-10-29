# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Number of top predictions to overlay on the frame.
DEFAULT_TOP_K = 3

# Fraction of the resized frame kept by the center crop, for models outside
# the 112x112 family (e.g. 256 -> 224).
CROP_RATIO = 0.875

# Clips sampled per video. 1 is a single fast inference; higher values sample
# the video more densely and average the results.
DEFAULT_NUM_CLIPS = 1

# ResNet-2Plus1D was trained on 15 fps clips with frames resized to 128x171
# before the 112x112 center crop, so its inputs must be built the same way.
KINETICS_SPATIAL_DIM = 112
KINETICS_TARGET_FPS = 15
KINETICS_RESIZE_H = 128
KINETICS_RESIZE_W = 171

# NOTE: exported models bake in both the per-channel mean/std normalization
# and the final softmax, so the app only needs to feed RGB frames scaled to
# [0, 1]; the output tensor is already per-class probabilities.
