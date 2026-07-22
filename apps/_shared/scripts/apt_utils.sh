#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Ubuntu/Linux apt package installation utilities.
#
# Functions:
#   install_apt_pkg <package> [extra_apt_args...]
#       Install an apt package if it is not already installed.
#   install_apt_pkgs <package> [<package> ...]
#       Install multiple apt packages, each idempotently.
#
# Usage: source apt_utils.sh
# ---------------------------------------------------------------------
_APT_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_APT_UTILS_DIR/load_versions.sh"
# shellcheck disable=SC1091
source "$_APT_UTILS_DIR/sudo.sh"
# shellcheck disable=SC1091
source "$_APT_UTILS_DIR/interactive.sh"
# shellcheck disable=SC1091
source "$_APT_UTILS_DIR/retry.sh"

_install_apt_pkg() {
    local pkg="$1"; shift
    if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q 'install ok installed'; then
        echo "::skip::${pkg}"
    else
        echo "::step::Installing ${pkg}"
        with_retry "apt-get install ${pkg}" -- $SUDO apt-get install -y "$pkg" "$@"
        echo "::done::${pkg}"
    fi
}

install_apt_pkg() {
    require_consent "Install apt package '$1' (uses sudo)" -- _install_apt_pkg "$@"
}

_install_apt_pkgs() {
    for pkg in "$@"; do
        _install_apt_pkg "$pkg"
    done
}

install_apt_pkgs() {
    require_consent "Install apt packages: $* (uses sudo)" -- _install_apt_pkgs "$@"
}
