# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Model input resolution (height, width). CavaFace expects 112x112 RGB.
INPUT_HEIGHT = 112
INPUT_WIDTH = 112

# Length of the face embedding vector the model emits.
EMBEDDING_DIM = 512

# Cosine-similarity threshold above which a detected face is treated as the same
# person as a gallery identity. Below it, the face is labeled "Unknown". Matches
# the qai_hub_models reference demo (similarity > 0.5 -> match). Because the Haar
# crop is not landmark-aligned, this may need on-device tuning via
# --recognition-threshold.
RECOGNITION_THRESHOLD = 0.5

# Label drawn for a detected face that matches no gallery identity.
UNKNOWN_LABEL = "Unknown"

# ---------------------------------------------------------------------
# Haar cascade face detector (OpenCV, CPU). Ships inside opencv-python-headless
# at cv2.data.haarcascades; no extra download.
# ---------------------------------------------------------------------
HAAR_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

# detectMultiScale parameters. SCALE_FACTOR is the image pyramid step,
# MIN_NEIGHBORS the overlap needed to keep a detection, MIN_SIZE the smallest
# face (in the detection-resolution frame) to report.
DETECTION_SCALE_FACTOR = 1.1
DETECTION_MIN_NEIGHBORS = 5
DETECTION_MIN_SIZE = (60, 60)

# Detection runs on the frame downscaled so its width is at most this many
# pixels (boxes are mapped back to full-frame coordinates). This is the main
# lever on detection cost for large camera frames.
DETECTION_MAX_WIDTH = 512

# Fraction by which each detected face box is expanded on every side before
# cropping, so the crop includes some context around the face (CavaFace was
# trained on slightly padded crops).
BOX_MARGIN = 0.2

# ---------------------------------------------------------------------
# Overlay drawing. Colors are RGB — frames are RGB in the pipeline and only
# flipped to BGR by webui.set_frame(frame[..., ::-1]) just before JPEG encoding.
# ---------------------------------------------------------------------
KNOWN_COLOR = (0, 255, 0)
UNKNOWN_COLOR = (255, 0, 0)
BOX_THICKNESS = 2

# ---------------------------------------------------------------------
# Default camera capture resolution (width, height), matching the sibling
# Ubuntu camera apps.
# ---------------------------------------------------------------------
VIDEO_WIDTH = 1024
VIDEO_HEIGHT = 768
