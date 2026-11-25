# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Directory (relative to the app root) holding the model file and its
# metadata.json
MODELS_DIR = "models"

# The supported models (DeepLab Xception, DeepLabV3+ MobileNet, FCN-ResNet50)
# are trained on the Pascal VOC label set (COCO_WITH_VOC_LABELS_V1): 20 object
# classes + background, for 21 classes total. The predicted class is the
# per-pixel argmax over the classes.
NUM_CLASSES = 21

# Class index treated as background. Background pixels are left untinted so the
# original frame shows through.
BACKGROUND_CLASS = 0

# Blend strength of the color overlay over foreground pixels, in [0, 1].
OVERLAY_ALPHA = 0.5
