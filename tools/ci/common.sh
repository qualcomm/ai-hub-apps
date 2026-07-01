# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Shared CI shell utilities.
#
# Functions:
#   repo_root
#       Print the absolute path to the repository root.
#   download_and_verify <url> <dest_file> [<sha256>]
#       Download <url> to <dest_file>. If <sha256> is provided, verifies the
#       checksum and exits non-zero if it does not match.
#
# Usage: source common.sh
# ---------------------------------------------------------------------

repo_root() {
    git rev-parse --show-toplevel
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
