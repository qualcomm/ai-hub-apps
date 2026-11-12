#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Fetches the per-device CDI and env auxiliary files an IMSDK app's container
# needs, pinned by URL in IMSDK_DEVICE_ASSETS_FILE (apps/_shared/scripts/versions.env).
#
# Functions:
#   imsdk_fetch_device_assets
#       Downloads the "env" and "cdi" files for $QAI_HUB_APPS_DEVICE_NAME /
#       $QAI_HUB_APPS_DEVICE_OS to the host paths named alongside their URL in
#       the device assets file, creating parent directories as needed. A file
#       already at its path with the pinned sha256 is left alone; a download
#       that does not match it is rejected.
#
# Usage: source imsdk_utils.sh
# ---------------------------------------------------------------------
_IMSDK_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_IMSDK_UTILS_DIR/sudo.sh"
# shellcheck disable=SC1091
source "$_IMSDK_UTILS_DIR/apt_utils.sh"
# shellcheck disable=SC1091
source "$_IMSDK_UTILS_DIR/retry.sh"
# shellcheck disable=SC1091
source "$_IMSDK_UTILS_DIR/interactive.sh"

_imsdk_device_assets_path() {
    export QAIHA_APP_ROOT="${QAIHA_APP_ROOT:?QAIHA_APP_ROOT must be set}"
    # shellcheck disable=SC1091
    source "$_IMSDK_UTILS_DIR/load_versions.sh"
    echo "$_IMSDK_UTILS_DIR/${IMSDK_DEVICE_ASSETS_FILE:?IMSDK_DEVICE_ASSETS_FILE is not set in versions.env}"
}

_imsdk_asset_field() {
    local assets_file="$1" device="$2" os="$3" kind="$4" field="$5"
    jq -er --arg device "$device" --arg os "$os" --arg kind "$kind" \
        '.[$device][$os][$kind].'"$field" "$assets_file"
}

_imsdk_sha256() {
    [ -f "$1" ] && sha256sum "$1" | cut -d' ' -f1
}

# Args: <kind> <url> <path> <sha256>, repeated.
_imsdk_write_assets() {
    while [ $# -ge 4 ]; do
        local kind="$1" url="$2" path="$3" sha256="$4"
        shift 4
        echo "::step::Fetching IMSDK $kind asset from $url to $path"
        local tmp
        tmp="$(mktemp)"
        with_retry --attempts 5 --backoff 5 "Download IMSDK $kind asset" -- \
            download_and_verify "$url" "$tmp" "$sha256" || { rm -f "$tmp"; return 1; }
        $SUDO mkdir -p "$(dirname "$path")"
        $SUDO install -m 644 "$tmp" "$path"
        rm -f "$tmp"
        echo "::done::$kind asset"
    done
}

imsdk_fetch_device_assets() {
    # IMSDK requires a CDI and an env file that must be present on the Host before a container runs.
    # This function pulls these files and puts them at the appropriate paths defined in IMSDK_DEVICE_ASSETS_FILE
    install_apt_pkgs jq curl

    local device="${QAI_HUB_APPS_DEVICE_NAME:?QAI_HUB_APPS_DEVICE_NAME must be set}"
    # A bare Ubuntu OS (e.g. "Ubuntu 24.04") means Ubuntu Desktop; its assets are
    # the desktop ones. Make server explicit in the OS string once it is supported.
    local os="${QAI_HUB_APPS_DEVICE_OS:?QAI_HUB_APPS_DEVICE_OS must be set}"
    local assets_file
    assets_file="$(_imsdk_device_assets_path)"
    if [ ! -f "$assets_file" ]; then
        echo "::error::IMSDK device assets file not found at '$assets_file'." >&2
        return 1
    fi

    local kind url path sha256
    local -a writes=() paths=()
    for kind in env cdi; do
        url="$(_imsdk_asset_field "$assets_file" "$device" "$os" "$kind" url)" || {
            echo "::error::No '$kind' asset for device '$device' / OS '$os' in '$assets_file'." >&2
            return 1
        }
        path="$(_imsdk_asset_field "$assets_file" "$device" "$os" "$kind" path)"
        sha256="$(_imsdk_asset_field "$assets_file" "$device" "$os" "$kind" sha256)"
        if [ "$(_imsdk_sha256 "$path")" = "$sha256" ]; then
            echo "::skip::IMSDK $kind asset at $path is up to date"
            continue
        fi
        writes+=("$kind" "$url" "$path" "$sha256")
        paths+=("$path")
    done
    [ ${#writes[@]} -gt 0 ] || return 0

    require_consent \
        "Write the IMSDK device files for '$device' ($os) to ${paths[*]}, overwriting any existing ones (uses sudo)" \
        -- _imsdk_write_assets "${writes[@]}" || return 1
}
