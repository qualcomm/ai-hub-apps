# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

# Model options
INPUT_HEIGHT = 256
INPUT_WIDTH = 256
NMS_IOU_THRESHOLD = 0.3
MIN_DETECTOR_BOX_SCORE = 0.95
MIN_LANDMARK_SCORE = 0.5
KEYPOINT_ROTATION_VEC_START_IDX = 0
KEYPOINT_ROTATION_VEC_END_IDX = 2
ROTATION_OFFSET_RADS = np.pi / 2
DETECT_BOX_OFFSET_XY = 0.5
DETECT_BOX_SCALE = 2.5
DETECTOR_SCORE_CLIPPING_THRESHOLD = 100

GESTURE_LABELS = [
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
]

HAND_LANDMARK_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
]
