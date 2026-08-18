#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/python_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/scripts/apt_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/scripts/pip_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/scripts/qairt_utils.sh"

install_python
install_qairt

$SUDO apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa
$SUDO apt-get update -q

install_apt_pkgs \
    libsndfile1 \
    ffmpeg \
    qcom-libdmabufheap

install_apt_pkg unzip

install_pip_deps --venv "$SCRIPT_DIR/.venv" -r "$SCRIPT_DIR/requirements.txt"
