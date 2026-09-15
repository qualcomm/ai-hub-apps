#!/usr/bin/env bash

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

USE_DOCKER=1
CLEAN=0
# shellcheck disable=SC2034
for arg in "$@"; do
    case "$arg" in
        --no-docker) USE_DOCKER=0 ;;
        --docker) USE_DOCKER=1 ;;
        --clean) CLEAN=1 ;;
        *) echo "::error::Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

echo "::skip::Nothing to build for mediapipe_hand_gesture_ubuntu_py; run it directly (see the app README)."
