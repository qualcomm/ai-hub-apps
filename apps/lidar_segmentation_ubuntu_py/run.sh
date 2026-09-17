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

SAMPLE_SCAN_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/salsanext/v3/000000.bin"
SAMPLE_SCAN="$SCRIPT_DIR/000000.bin"

activate_venv

# There is no host capture device to read a scan from, so a sample stands in.
if [[ "$*" != *--lidar-source* ]]; then
    [ -f "$SAMPLE_SCAN" ] || wget -q -O "$SAMPLE_SCAN" "$SAMPLE_SCAN_URL"
    set -- --lidar-source "$SAMPLE_SCAN" "$@"
fi

exec python main.py --qairt-path "$QAIRT_PATH" "$@"
