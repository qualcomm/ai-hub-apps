#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source ../_shared/scripts/pip_utils.sh

TEST_VIDEO_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/object_detection_3d_ubuntu_py/test/traffic.mp4"
TEST_VIDEO="$SCRIPT_DIR/traffic.mp4"
MODEL="$SCRIPT_DIR/models/detector.tflite"
LABELS="$SCRIPT_DIR/models/labels.txt"
OUT_VIDEO="$(mktemp --suffix=.mp4)"

for f in "$MODEL" "$LABELS"; do
    if [ ! -f "$f" ]; then
        echo "::error::$(basename "$f") not found at $f. Fetch the model first." >&2
        exit 1
    fi
done

activate_venv

curl -fsSL -o "$TEST_VIDEO" "$TEST_VIDEO_URL"

echo "Running object detection on $TEST_VIDEO"
rc=0
output="$(python main.py \
    --input-config "$TEST_VIDEO" \
    --output file \
    --output-file "$OUT_VIDEO" \
    "$@" 2>&1)" || rc=$?
echo "$output"
if [ "$rc" -ne 0 ]; then
    echo "::error::main.py exited with code $rc." >&2
    exit 1
fi

# main.py prints "Number of objects: <n>" once the pipeline ends. Require at
# least one detection across the video.
objects="$(echo "$output" | sed -nE 's/^Number of objects: *([0-9]+)$/\1/p' | tail -n1)"
if [ -z "$objects" ]; then
    echo "::error::Could not find object count in output." >&2
    exit 1
fi
if [ "$objects" -lt 1 ]; then
    echo "::error::No objects detected in $TEST_VIDEO." >&2
    exit 1
fi

if [ ! -s "$OUT_VIDEO" ]; then
    echo "::error::Annotated output video $OUT_VIDEO is missing or empty." >&2
    exit 1
fi

echo "Object detection test passed ($objects objects)."
