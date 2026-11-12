#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Retry wrapper for flaky commands.
#
# Re-runs a command with exponential backoff until it succeeds or the
# attempt budget is exhausted.
#
# Functions:
#   with_retry [--attempts N] [--backoff BASE] <description> -- <command> [args...]
#       Run <command>, retrying on failure.
#       --attempts N  : total attempts (default 3).
#       --backoff BASE: base backoff seconds (default 5). After a failed
#                       attempt, sleeps BASE * 2^(attempt-1): 5s, 10s, 15s, ...
#       - success -> return 0 immediately.
#       - failure with attempts remaining -> report, sleep, retry.
#       - all attempts fail -> return the command's exit code so callers
#                              under `set -e` still abort.
#   download_and_verify <url> <dest_file> [<sha256>]
#       Download <url> to <dest_file>. If <sha256> is provided, verifies the
#       checksum and exits non-zero if it does not match.
#
# Usage: source retry.sh
# ---------------------------------------------------------------------

with_retry() {
    local attempts=3
    local backoff=5

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --attempts) attempts="$2"; shift 2 ;;
            --backoff)  backoff="$2";  shift 2 ;;
            *) break ;;
        esac
    done

    local description="$1"; shift
    if [ "${1:-}" = "--" ]; then
        shift
    fi

    local attempt=1
    local rc=0
    while true; do
        rc=0
        "$@" || rc=$?
        if [ "$rc" -eq 0 ]; then
            return 0
        fi

        if [ "$attempt" -ge "$attempts" ]; then
            echo "::error::${description} failed after ${attempts} attempt(s) (exit ${rc})" >&2
            return "$rc"
        fi

        local delay=$((backoff * (1 << (attempt - 1))))
        echo "::step::${description} failed (exit ${rc}); retrying in ${delay}s (attempt $((attempt + 1))/${attempts})" >&2
        sleep "$delay"
        attempt=$((attempt + 1))
    done
}

download_and_verify() {
    local url="$1"
    local dest="$2"
    local sha256="${3:-}"

    echo "Downloading $(basename "$dest")"
    echo "   URL: $url"
    # Explicit returns: callers in a `||` or `if` context run without `set -e`.
    curl -fSL --max-time 120 -o "$dest" "$url" || return 1
    if [ -n "$sha256" ]; then
        echo "$sha256  $dest" | sha256sum -c - || return 1
    fi
    echo "Downloaded and verified"
}
