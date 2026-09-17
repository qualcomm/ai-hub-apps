#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source ../_shared/scripts/pip_utils.sh
source ../_shared/scripts/qairt_utils.sh

TEST_SCAN_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/salsanext/v3/000000.bin"
TEST_SCAN="$SCRIPT_DIR/000000.bin"

activate_venv

wget -q -O "$TEST_SCAN" "$TEST_SCAN_URL"

python main.py \
    --lidar-source "$TEST_SCAN" \
    --output "$SCRIPT_DIR/000000.png" \
    --qairt-path "$QAIRT_PATH" \
    "$@"
