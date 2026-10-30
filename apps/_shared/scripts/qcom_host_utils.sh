#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Qualcomm host (driver) package installation for Ubuntu devices.
#
# These packages live on the host, not in the app container: they provide the
# Adreno/FastRPC/QNN user-space drivers that talk to the kernel.
#
# Functions:
#   qcom_host_libcdsprpc
#       Echo the path to libcdsprpc.so, or nothing if it is not installed.
#   install_qcom_host_pkgs
#       Install the host packages after a single user consent prompt. A no-op
#       when libcdsprpc.so is already present. Errors on non-arm64 hosts.
#
# Usage: source qcom_host_utils.sh
# ---------------------------------------------------------------------
_QCOM_HOST_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_QCOM_HOST_UTILS_DIR/sudo.sh"
# shellcheck disable=SC1091
source "$_QCOM_HOST_UTILS_DIR/apt_utils.sh"
# shellcheck disable=SC1091
source "$_QCOM_HOST_UTILS_DIR/interactive.sh"
# shellcheck disable=SC1091
source "$_QCOM_HOST_UTILS_DIR/retry.sh"

QCOM_HOST_PPA="ppa:ubuntu-qcom-iot/qcom-ppa"
QCOM_HOST_PKGS=(qcom-adreno1 qcom-fastrpc1 qcom-fastrpc-dev libqnn1)

qcom_host_libcdsprpc() {
    local candidate
    for candidate in /usr/lib/aarch64-linux-gnu/libcdsprpc.so /usr/lib/libcdsprpc.so; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

_install_qcom_host_pkgs() {
    echo "::step::Adding $QCOM_HOST_PPA"
    with_retry "apt-add-repository $QCOM_HOST_PPA" -- \
        $SUDO apt-add-repository -y "$QCOM_HOST_PPA"
    with_retry "apt-get update" -- $SUDO apt-get update -q
    echo "::done::$QCOM_HOST_PPA"

    install_apt_pkgs "${QCOM_HOST_PKGS[@]}"
}

install_qcom_host_pkgs() {
    if qcom_host_libcdsprpc >/dev/null; then
        echo "::skip::Qualcomm host packages (libcdsprpc.so already installed)"
        return 0
    fi

    local arch
    arch="$(uname -m)"
    if [ "$arch" != "aarch64" ]; then
        echo "::error::libcdsprpc.so not found, and this host is '$arch', not aarch64. The Qualcomm host packages are arm64-only." >&2
        return 1
    fi

    require_consent \
        "Install the Qualcomm host packages ${QCOM_HOST_PKGS[*]} from $QCOM_HOST_PPA (uses sudo)" \
        -- _install_qcom_host_pkgs

    if ! qcom_host_libcdsprpc >/dev/null; then
        echo "::error::libcdsprpc.so is still missing after installing ${QCOM_HOST_PKGS[*]}." >&2
        return 1
    fi
}
