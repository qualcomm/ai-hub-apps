# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Model input resolution (height, width). Posenet MobileNet expects 513x257 RGB.
INPUT_HEIGHT = 513
INPUT_WIDTH = 257

# The model emits heatmaps/offsets/displacements on a grid downsampled by this
# factor relative to the input resolution.
OUTPUT_STRIDE = 16

# Decode parameters.
MAX_POSE_DETECTIONS = 10
SCORE_THRESHOLD = 0.25
NMS_RADIUS = 20
MIN_POSE_SCORE = 0.25
MIN_PART_SCORE = 0.1

PART_NAMES = [
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
]

NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}

# Edges traversed when decoding a pose from a root keypoint (a minimum spanning
# tree over the keypoints).
POSE_CHAIN = [
    ("nose", "leftEye"),
    ("leftEye", "leftEar"),
    ("nose", "rightEye"),
    ("rightEye", "rightEar"),
    ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"),
    ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"),
    ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
]

PARENT_CHILD_TUPLES = [
    (PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN
]

# Edges drawn as the skeleton overlay.
CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"),
    ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"),
    ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"),
    ("leftHip", "rightHip"),
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]
