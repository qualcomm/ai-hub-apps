# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Shared CI shell utilities.
#
# Functions:
#   repo_root
#       Print the absolute path to the repository root.
#   get_version <KEY>
#       Print the value of <KEY> from tools/versions.env (KEY="VALUE" lines).
#       Exits non-zero if the key is not found.
#   download_and_verify <url> <dest_file> [<sha256>]
#       Download <url> to <dest_file>. If <sha256> is provided, verifies the
#       checksum and exits non-zero if it does not match.
#
# Usage: source common.sh
# ---------------------------------------------------------------------

_COMMON_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

repo_root() {
    git rev-parse --show-toplevel
}

get_version() {
    local key="$1"
    local versions_file="$_COMMON_SH_DIR/../versions.env"
    local value
    value="$(
        # shellcheck source=tools/versions.env
        source "$versions_file"
        printf '%s' "${!key:-}"
    )"
    if [ -z "$value" ]; then
        echo "error: version key '$key' not found in $versions_file" >&2
        return 1
    fi
    printf '%s\n' "$value"
}

download_and_verify() {
    local url="$1"
    local dest="$2"
    local sha256="${3:-}"

    echo "Downloading $(basename "$dest")"
    echo "   URL: $url"
    curl -fSL --max-time 120 -o "$dest" "$url"
    if [ -n "$sha256" ]; then
        echo "$sha256  $dest" | sha256sum -c -
    fi
    echo "Downloaded and verified"
}
