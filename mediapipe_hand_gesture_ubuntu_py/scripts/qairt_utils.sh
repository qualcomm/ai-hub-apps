#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Ubuntu/Linux QAIRT SDK installation utilities.
#
# Functions:
#   install_qairt [--force]
#       Download and extract the QAIRT SDK to $QAIRT_PATH. Skips if already
#       installed unless --force is passed. After sourcing this script,
#       $QAIRT_PATH holds the SDK root directory.
#
# Usage: source qairt_utils.sh
# ---------------------------------------------------------------------
_QAIRT_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_QAIRT_UTILS_DIR/load_versions.sh"
# shellcheck disable=SC1091
source "$_QAIRT_UTILS_DIR/apt_utils.sh"

QAIRT_ROOT="/opt/qcom/aistack/qairt"
QAIRT_PATH="${QAIRT_ROOT}/${QAIRT_SDK_FULL_VERSION}"
export QAIRT_ROOT QAIRT_PATH

install_qairt() {
    if [ "${QAIRT_INSTALL_SKIP:-}" = "true" ]; then
        echo "::skip::QAIRT SDK install deferred (QAIRT_INSTALL_SKIP=true)"
        return 0
    fi

    local force=0
    if [ "${1:-}" = "--force" ]; then force=1; fi

    if [ -d "$QAIRT_PATH" ] && [ -n "$(ls -A "$QAIRT_PATH" 2>/dev/null)" ] && [ "$force" -eq 0 ]; then
        echo "::skip::QAIRT SDK already installed at $QAIRT_PATH"
        return 0
    fi

    $SUDO apt-get update -q
    install_apt_pkg unzip

    local url="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QAIRT_SDK_FULL_VERSION}/v${QAIRT_SDK_FULL_VERSION}.zip"
    local tmp_zip
    tmp_zip="$(mktemp --suffix=.zip)"

    echo "::step::Downloading QAIRT SDK ${QAIRT_SDK_FULL_VERSION}"
    echo "   URL: $url"
    wget -q -O "$tmp_zip" "$url"
    echo "::done::download"

    echo "::step::Extracting QAIRT SDK"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    unzip -q "$tmp_zip" -d "$tmp_dir"
    rm "$tmp_zip"

    $SUDO mkdir -p "$QAIRT_ROOT"
    $SUDO rm -rf "$QAIRT_PATH"
    $SUDO mv "$tmp_dir/qairt/${QAIRT_SDK_FULL_VERSION}" "$QAIRT_PATH"
    rm -rf "$tmp_dir"
    echo "::done::QAIRT SDK installed at $QAIRT_PATH"
}
