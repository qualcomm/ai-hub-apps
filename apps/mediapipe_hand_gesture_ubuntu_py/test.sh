#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source ../_shared/scripts/qairt_utils.sh

TEST_VIDEO_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/mediapipe_hand_gesture_ubuntu_py/test/gesture.MP4"
TEST_VIDEO="$SCRIPT_DIR/gesture.mp4"

if [ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    echo "error: virtual environment not found. Run install_runtime.sh first." >&2
    exit 1
fi
source "$SCRIPT_DIR/.venv/bin/activate"

wget -q -O "$TEST_VIDEO" "$TEST_VIDEO_URL"

# Force decodebin to use the installed libav software decoder.
export GST_PLUGIN_FEATURE_RANK="v4l2h264dec:0,qtivdec:0,qtivdechw:0"

python main.py \
    --video-gstreamer-source "filesrc location=$TEST_VIDEO ! decodebin" \
    --qairt-path "$QAIRT_PATH"
