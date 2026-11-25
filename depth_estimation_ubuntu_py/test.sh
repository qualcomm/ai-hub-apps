#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/pip_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/scripts/qairt_utils.sh"

ASSET_BASE="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/depth_estimation_ubuntu_py"
TEST_IMAGE="$SCRIPT_DIR/image.jpg"
TEST_IMAGE_SHA256="594f989e4d5a64afeb7eda689ca255546ca3bf75dd9ee51899d81010950ee5e8"
REFERENCE_IMAGE="$SCRIPT_DIR/test_output.png"
OUTPUT_IMAGE="$SCRIPT_DIR/test_output_actual.png"

activate_venv

wget -q -O "$TEST_IMAGE" "$ASSET_BASE/image.jpg"
echo "$TEST_IMAGE_SHA256  $TEST_IMAGE" | sha256sum -c -
wget -q -O "$REFERENCE_IMAGE" "$ASSET_BASE/test_output.png"

rm -f "$OUTPUT_IMAGE"

python main.py \
    --video-gstreamer-source "filesrc location=$TEST_IMAGE ! decodebin ! imagefreeze num-buffers=120" \
    --video-source-width 640 \
    --video-source-height 640 \
    --qairt-path "$QAIRT_PATH" \
    --auto-capture-after-frames 5 \
    --output-image "$OUTPUT_IMAGE" \
    "$@"

python compare_images.py "$OUTPUT_IMAGE" "$REFERENCE_IMAGE" --threshold 10
