#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source ../_shared/scripts/pip_utils.sh
source ../_shared/scripts/qairt_utils.sh

TEST_VIDEO_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/video_classification_ubuntu_py/surfing_cutback.mp4"
TEST_VIDEO="$SCRIPT_DIR/surfing_cutback.mp4"
OUTPUT_IMAGE="$SCRIPT_DIR/test_output_actual.jpg"
EXPECTED="surfing water"

activate_venv

wget -q -O "$TEST_VIDEO" "$TEST_VIDEO_URL"

rm -f "$OUTPUT_IMAGE"

# Sampling 5 clips puts the correct class at top-1 on this video; a single
# clip leaves it at rank 2.
echo "=== Classifying $(basename "$TEST_VIDEO") (expecting '$EXPECTED') ==="
output="$(python main.py \
    --video "$TEST_VIDEO" \
    --qairt-path "$QAIRT_PATH" \
    --output "$OUTPUT_IMAGE" \
    --num-clips 5 \
    "$@")"
echo "$output"

if ! echo "$output" | grep -qiF "$EXPECTED"; then
    echo "FAIL: '$EXPECTED' not among the top predictions." >&2
    exit 1
fi
echo "PASS: predicted '$EXPECTED'."

if [ ! -s "$OUTPUT_IMAGE" ]; then
    echo "FAIL: no annotated output written to $OUTPUT_IMAGE" >&2
    exit 1
fi
echo "PASS: wrote annotated output."
