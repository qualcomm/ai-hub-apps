#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/python_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/scripts/apt_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/scripts/pip_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/scripts/qairt_utils.sh"

install_python
install_qairt

$SUDO apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa
$SUDO apt-get update -q

install_apt_pkgs \
    libcairo2-dev \
    pkg-config \
    libgirepository1.0-dev \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-qcom-qmmfsrc \
    gstreamer1.0-plugins-qcom-vtransform \
    v4l2loopback-dkms \
    v4l2loopback-utils \
    gstreamer1.0-libav

install_apt_pkg unzip

install_pip_deps --venv "$SCRIPT_DIR/.venv" -r "$SCRIPT_DIR/requirements.txt"
