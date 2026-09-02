#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/qairt_utils.sh"

# The app is demoed on a still image hosted with the app's assets. A still
# image through `filesrc ! decodebin` emits a single buffer then EOS, so
# `imagefreeze` is appended to repeat that frame. `num-buffers` bounds the run
# to a finite number of frames so the pipeline reaches EOS and the app exits
# cleanly (otherwise imagefreeze would stream forever and the test would hang).
TEST_IMAGE_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/semantic_segmentation_ubuntu_py/test/demo_image.png"
TEST_IMAGE="$SCRIPT_DIR/demo_image.png"

if [ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    echo "error: virtual environment not found. Run install_runtime.sh first." >&2
    exit 1
fi
source "$SCRIPT_DIR/.venv/bin/activate"

wget -q -O "$TEST_IMAGE" "$TEST_IMAGE_URL"

python main.py \
    --video-gstreamer-source "filesrc location=$TEST_IMAGE ! decodebin ! imagefreeze num-buffers=120" \
    --qairt-path "$QAIRT_PATH"
