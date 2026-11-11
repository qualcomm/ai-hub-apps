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

ASSET_BASE="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/colourization_ubuntu_py"
TEST_IMAGE="$SCRIPT_DIR/ddcolor_sample.jpeg"
REFERENCE_IMAGE="$SCRIPT_DIR/output_image.png"
OUTPUT_IMAGE="$SCRIPT_DIR/output_image_actual.png"

activate_venv

wget -q -O "$TEST_IMAGE" "$ASSET_BASE/ddcolor_sample.jpeg"
wget -q -O "$REFERENCE_IMAGE" "$ASSET_BASE/output_image.png"

rm -f "$OUTPUT_IMAGE"

python main.py \
    --video-gstreamer-source "filesrc location=$TEST_IMAGE ! decodebin ! imagefreeze num-buffers=120" \
    --qairt-path "$QAIRT_PATH" \
    --auto-capture-after-frames 5 \
    --output-image "$OUTPUT_IMAGE" \
    "$@"

# The reference PNG was saved with red/blue swapped (BGR array written as RGB), so
# it is channel-reversed.
python compare_images.py "$OUTPUT_IMAGE" "$REFERENCE_IMAGE" \
    --threshold 5 \
    --swap-reference-channels
