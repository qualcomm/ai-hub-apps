# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

MODELS_DIR = "models"

# In prediction-index order; index 0 is the ignored class.
CLASS_NAMES = (
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
)
NUM_CLASSES = len(CLASS_NAMES)

# The SemanticKITTI palette as RGB, one color per CLASS_NAMES entry.
CLASS_COLORS = (
    (0, 0, 0),  # unlabeled
    (100, 150, 245),  # car
    (100, 230, 245),  # bicycle
    (30, 60, 150),  # motorcycle
    (80, 30, 180),  # truck
    (0, 0, 255),  # other-vehicle
    (255, 30, 30),  # person
    (255, 40, 200),  # bicyclist
    (150, 30, 90),  # motorcyclist
    (255, 0, 255),  # road
    (255, 150, 255),  # parking
    (75, 0, 75),  # sidewalk
    (175, 0, 75),  # other-ground
    (255, 200, 0),  # building
    (255, 120, 50),  # fence
    (0, 175, 0),  # vegetation
    (135, 60, 0),  # trunk
    (150, 240, 80),  # terrain
    (255, 240, 150),  # pole
    (255, 0, 0),  # traffic-sign
)

# A scan record is x, y, z, remission; the projection prepends range.
SCAN_FIELDS = 4
NUM_INPUT_CHANNELS = 5

# Vertical field of view of the HDL-64E the model was trained on, in degrees.
FOV_UP_DEG = 3.0
FOV_DOWN_DEG = -25.0

# Per-channel training statistics, ordered as the input channels.
INPUT_MEANS = (12.12, 10.88, 0.23, -1.04, 0.21)
INPUT_STDS = (12.32, 11.47, 6.91, 0.86, 0.16)

# Half-extent of the rendered ground plane, in meters.
BEV_RANGE_M = 50.0
BEV_SIZE_PX = 640

RANGE_VIEW_WIDTH_PX = 1024
RANGE_VIEW_HEIGHT_PX = 128
