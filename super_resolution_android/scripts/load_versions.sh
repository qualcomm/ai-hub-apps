#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Loads versions.env (KEY=VALUE) into the current shell environment.
# All variables are exported so sub-processes inherit them.
#
# Usage: source load_versions.sh
# ---------------------------------------------------------------------
_LOAD_VERSIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VERSIONS_FILE="$_LOAD_VERSIONS_DIR/versions.env"

if [ ! -f "$_VERSIONS_FILE" ]; then
    echo "error: versions.env not found at $_VERSIONS_FILE" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$_VERSIONS_FILE"
set +a
