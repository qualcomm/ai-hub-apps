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

TEST_IMAGE_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/easyocr_ubuntu_py/test/gracewood.png"
TEST_IMAGE="$SCRIPT_DIR/gracewood.png"

REFERENCE_TEXT="$SCRIPT_DIR/gracewood_output.txt"
OUTPUT_TEXT="$SCRIPT_DIR/gracewood_output_actual.txt"
printf 'GRACEWOOD LN\nPRIVATE ROAD\n' > "$REFERENCE_TEXT"

activate_venv

wget -q -O "$TEST_IMAGE" "$TEST_IMAGE_URL"

# --auto-capture-after-frames stands in for pressing 's'. The source size matches
# TEST_IMAGE (371x108) so the image is not rescaled before detection. The
# recognizer runs on CPU, so a UserWarning about the HTP delegate is expected.
python main.py \
    --video-gstreamer-source "filesrc location=$TEST_IMAGE ! decodebin ! imagefreeze num-buffers=120" \
    --video-source-width 371 \
    --video-source-height 108 \
    --qairt-path "$QAIRT_PATH" \
    --output-text "$OUTPUT_TEXT" \
    --auto-capture-after-frames 5 \
    "$@"

if diff -u "$REFERENCE_TEXT" "$OUTPUT_TEXT"; then
    echo "PASS: predicted text matches."
else
    echo "FAIL: predicted text differs." >&2
    exit 1
fi
