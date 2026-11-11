#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# install_runtime.sh is skipped once it has succeeded for this exact app content;
# the marker holds a hash of everything the install depends on.
#
# Functions:
#   runtime_stamp <file>...
#       Echo a hash of the given files that exist.
#   install_runtime_native <marker> <stamp>
#       Run install_runtime.sh unless <marker> already holds <stamp>.
#   install_runtime_in_container <container> <marker> <stamp> [<docker exec arg>...]
#       Same, inside <container>, with the marker at <marker> in the container.
#
# Usage: source runtime_utils.sh
# ---------------------------------------------------------------------

runtime_stamp() {
    local files=()
    local f
    for f in "$@"; do
        if [ -f "$f" ]; then
            files+=("$f")
        fi
    done
    sha1sum -- "${files[@]}" | sha1sum | cut -d' ' -f1
}

# Explain why the runtime is (re)installed: no marker yet, or a stale one.
log_runtime_marker_miss() {
    local marker="$1" installed="$2" stamp="$3"
    if [ -z "$installed" ]; then
        echo "No runtime marker at $marker; installing runtime"
    else
        echo "Runtime marker at $marker is stale (recorded $installed, current $stamp); reinstalling runtime"
    fi
}

install_runtime_native() {
    local marker="$1" stamp="$2" installed
    installed="$(cat "$marker" 2>/dev/null || true)"
    if [ "$installed" = "$stamp" ]; then
        echo "::skip::Runtime already installed (marker at $marker matches $stamp)"
    else
        log_runtime_marker_miss "$marker" "$installed" "$stamp"
        echo "::step::Installing runtime"
        bash install_runtime.sh
        printf '%s\n' "$stamp" > "$marker"
        echo "Wrote runtime marker $stamp to $marker"
        echo "::done::Installing runtime"
    fi
}

install_runtime_in_container() {
    local container="$1" marker="$2" stamp="$3" installed
    shift 3
    installed="$(${SUDO:-} docker exec "$container" cat "$marker" 2>/dev/null || true)"
    if [ "$installed" = "$stamp" ]; then
        echo "::skip::Runtime already installed in $container (marker at $marker matches $stamp)"
    else
        log_runtime_marker_miss "$container:$marker" "$installed" "$stamp"
        echo "::step::Installing runtime in $container"
        ${SUDO:-} docker exec "$@" -w /app "$container" bash install_runtime.sh
        ${SUDO:-} docker exec "$container" bash -c \
            'mkdir -p "$(dirname "$1")" && printf "%s\n" "$2" > "$1"' _ "$marker" "$stamp"
        echo "Wrote runtime marker $stamp to $container:$marker"
        echo "::done::Installing runtime"
    fi
}
