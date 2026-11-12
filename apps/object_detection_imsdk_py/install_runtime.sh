#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source ../_shared/scripts/pip_utils.sh

# The IMSDK image's system python3 owns qimsdk, PyGObject and numpy, and is
# externally managed, so the venv is created with --system-site-packages to keep
# them importable.
create_venv --python python3 -- --system-site-packages
activate_venv

echo "::step::Installing Python dependencies"
python -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
echo "::done::pip install"
