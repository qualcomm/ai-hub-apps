#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ../_shared/scripts/pip_utils.sh
source ../_shared/scripts/qairt_utils.sh

# Two public reference photos of the same person. Image 1 is enrolled as a
# gallery identity; recognizing image 2 should report that same identity.
TEST_ASSET_BASE="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/face_recognition_ubuntu_py/test"
IMAGE_1_NAME="cavaface_demo_input_1.jpg"
IMAGE_2_NAME="cavaface_demo_input_2.jpg"
IMAGE_1="$SCRIPT_DIR/$IMAGE_1_NAME"
IMAGE_2="$SCRIPT_DIR/$IMAGE_2_NAME"

activate_venv

wget -q -O "$IMAGE_1" "$TEST_ASSET_BASE/$IMAGE_1_NAME"
wget -q -O "$IMAGE_2" "$TEST_ASSET_BASE/$IMAGE_2_NAME"

# Build a one-identity gallery from image 1.
GALLERY_DIR="$SCRIPT_DIR/test_gallery"
IDENTITY="reference_person"
rm -rf "$GALLERY_DIR"
mkdir -p "$GALLERY_DIR/$IDENTITY"
cp "$IMAGE_1" "$GALLERY_DIR/$IDENTITY/"

echo "=== Recognizing $IMAGE_2_NAME against gallery (expecting $IDENTITY) ==="
output="$(python main.py \
    --image "$IMAGE_2" \
    --gallery-dir "$GALLERY_DIR" \
    --qairt-path "$QAIRT_PATH")"
echo "$output"

# The app must run face recognition on the NPU. A CPU fallback (missing/failed
# QNN delegate) is a test failure, not a silent pass, so require the NPU backend
# marker that main.py prints on successful delegate initialization.
if ! echo "$output" | grep -qF "Backend: QNN NPU delegate (Hexagon)"; then
    echo "FAIL: model did not run on the NPU (CPU fallback detected)" >&2
    exit 1
fi

# Match the per-face result line ("reference_person: 0.87"), not the enrollment
# summary line, so an "Unknown" result correctly fails the test.
if echo "$output" | grep -qE "^${IDENTITY}: "; then
    echo "PASS: face recognized as $IDENTITY"
else
    echo "FAIL: face not recognized as $IDENTITY" >&2
    exit 1
fi
