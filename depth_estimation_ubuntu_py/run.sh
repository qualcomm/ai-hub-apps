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

activate_venv

# With no --video-device, main.py defaults to the host's first working v4l2
# camera via get_default_video_device().
exec python main.py --qairt-path "$QAIRT_PATH" "$@"
