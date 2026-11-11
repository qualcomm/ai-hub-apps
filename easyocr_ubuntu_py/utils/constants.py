# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""EasyOCR constants: CRAFT detector and CTC recognizer settings."""

# --- Detector (CRAFT) ---
# Score maps are downsampled by this factor relative to the detector input.
DETECTOR_NET_RATIO = 2

# CRAFT box-decoding thresholds (EasyOCR DETECTOR_ARGS).
TEXT_THRESHOLD = 0.7
LINK_THRESHOLD = 0.4
LOW_TEXT = 0.4

# group_text_box() parameters (EasyOCR DETECTOR_ARGS).
SLOPE_THS = 0.1
YCENTER_THS = 0.5
HEIGHT_THS = 0.5
WIDTH_THS = 0.5
ADD_MARGIN = 0.1
MIN_SIZE = 20

# --- Recognizer (CTC) ---
# "english_g2" character set. Order must match the model's output classes: the
# CTC blank is class 0, so class i (i >= 1) maps to CHARSET[i - 1].
CHARSET = (
    "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)
