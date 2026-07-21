#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Ubuntu/Linux Python installation utilities.
#
# Functions:
#   install_python
#       Install the Python version given by $PYTHON_VERSION (default from
#       versions.env) via apt. Also installs python3-venv and pip, then
#       bootstraps uv for fast package management.
#
# Override the version before sourcing or before calling:
#   PYTHON_VERSION=3.12
#   install_python
#
# Usage: source python_utils.sh
# ---------------------------------------------------------------------
_PYTHON_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_PYTHON_UTILS_DIR/load_versions.sh"
# shellcheck disable=SC1091
source "$_PYTHON_UTILS_DIR/apt_utils.sh"

install_python() {
    local ver="${PYTHON_VERSION}"

    echo "::step::Adding deadsnakes PPA for python${ver}"
    install_apt_pkg software-properties-common
    $SUDO add-apt-repository -y ppa:deadsnakes/ppa
    $SUDO apt-get update -q
    echo "::done::deadsnakes PPA"

    install_apt_pkg "python${ver}"
    install_apt_pkg "python${ver}-venv"
    install_apt_pkg "python${ver}-dev"
    install_apt_pkg python3-pip

    echo "::step::Bootstrapping uv for python${ver}"
    "python${ver}" -m pip install --quiet uv
    echo "::done::uv"
}
