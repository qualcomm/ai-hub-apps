#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source ../_shared/scripts/pip_utils.sh

activate_venv

# USB camera (the host's first one, unless -i /dev/videoN) -> detection -> RTSP,
# played in the browser at http://<device-ip>:8080.
exec python main.py --input-type usb-camera --output rtsp "$@"
