#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Loads versions.env (KEY=VALUE) into the current shell environment.
# All variables are exported so sub-processes inherit them.
# If $QAIHA_APP_ROOT/versions.override.env exists, its keys are layered on top
# of the global versions.env (the override wins).
#
# Usage: source load_versions.sh
# ---------------------------------------------------------------------
_LOAD_VERSIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VERSIONS_FILE="$_LOAD_VERSIONS_DIR/versions.env"

if [ -z "${QAIHA_APP_ROOT:-}" ]; then
    echo "error: QAIHA_APP_ROOT is required to use this utility. Set using 'export QAIHA_APP_ROOT=<app dir>'" >&2
    exit 1
fi

if [ ! -f "$_VERSIONS_FILE" ]; then
    echo "error: versions.env not found at $_VERSIONS_FILE" >&2
    exit 1
fi

for _versions_file in "$_VERSIONS_FILE" "$QAIHA_APP_ROOT/versions.override.env"; do
    [ -f "$_versions_file" ] || continue
    set -a
    # shellcheck disable=SC1090
    source "$_versions_file"
    set +a
done
unset _versions_file
