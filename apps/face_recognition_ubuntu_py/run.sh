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

activate_venv

# Seed a one-identity gallery from a public reference photo so the live demo has
# someone to recognize out of the box. Override with --gallery-dir <dir> to use
# your own enrolled identities (see README for the directory layout).
TEST_ASSET_BASE="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/face_recognition_ubuntu_py/test"
IMAGE_1_NAME="cavaface_demo_input_1.jpg"
IMAGE_1="$SCRIPT_DIR/$IMAGE_1_NAME"
GALLERY_DIR="$SCRIPT_DIR/gallery"
IDENTITY="reference_person"

if [ ! -d "$GALLERY_DIR/$IDENTITY" ]; then
    wget -q -O "$IMAGE_1" "$TEST_ASSET_BASE/$IMAGE_1_NAME"
    mkdir -p "$GALLERY_DIR/$IDENTITY"
    cp "$IMAGE_1" "$GALLERY_DIR/$IDENTITY/"
fi

# With no --video-device, main.py defaults to the host's first working v4l2
# camera via get_default_video_device().
exec python main.py --gallery-dir "$GALLERY_DIR" --qairt-path "$QAIRT_PATH" "$@"
