# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Directory (relative to the app root) holding the model files, their
# metadata.json, and labels.txt.
MODELS_DIR = "models"
LABELS_FILE = "labels.txt"

# Output tensor names that identify each component of the model bundle. Used
# instead of file names so a renamed asset still resolves.
DETECTOR_OUTPUTS = ("boxes", "scores", "class_idx")
BOX3D_OUTPUTS = ("orient", "conf", "dim")

NMS_SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.3

# Per-frame cap on how many detections are sent through the 3D head, highest
# score first. Each one costs an extra inference.
MAX_DETECTIONS = 5

# Horizontal field of view (degrees) of the camera, used to build the projection
# matrix that the 3D box geometry is solved in. Default approximates the KITTI
# camera the model was trained on.
DEFAULT_HFOV_DEG = 82.0

# Number of orientation bins the 3D head predicts (cos, sin) for.
NUM_ORIENTATION_BINS = 2

# COCO label -> KITTI class. Labels not listed here have no average dimension
# and are skipped.
COCO_TO_KITTI = {
    "person": "pedestrian",
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
    "motorbike": "cyclist",
    "car": "car",
    "bus": "truck",
    "truck": "truck",
    "train": "tram",
}

# Mean KITTI object dimensions (height, width, length) in metres. The 3D head
# predicts a residual on top of these.
CLASS_AVERAGE_DIMS = {
    "car": (1.526083432, 1.628589868, 3.883954492),
    "cyclist": (1.737203442, 0.596773202, 1.763546404),
    "misc": (1.907132580, 1.513833505, 3.575590956),
    "pedestrian": (1.760706485, 0.660189436, 0.842284377),
    "person_sitting": (1.274954955, 0.594909910, 0.802027027),
    "tram": (3.528923679, 2.543737769, 16.094266145),
    "truck": (3.251709324, 2.585091408, 10.109076782),
    "van": (2.206592313, 1.902079616, 5.078366507),
}

# RGB colors and line width for the projected 3D box.
BOX_COLOR = (0, 255, 0)
BOX_FRONT_COLOR = (0, 0, 255)
BOX_THICKNESS = 1
LABEL_COLOR = (255, 255, 0)
