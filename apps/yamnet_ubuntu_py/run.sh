#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source ../_shared/scripts/qairt_utils.sh

if [ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    echo "error: virtual environment not found. Run install_runtime.sh first." >&2
    exit 1
fi
source "$SCRIPT_DIR/.venv/bin/activate"

# With no --audio-file, main.py classifies live audio from the default microphone.
exec python main.py --qairt-path "$QAIRT_PATH" "$@"
