#!/usr/bin/env bash

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$APP_DIR"
cd "$APP_DIR"

USE_DOCKER=1
CLEAN=0
RUN_TEST=0
APP_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-docker) USE_DOCKER=0 ;;
        --docker) USE_DOCKER=1 ;;
        --clean) CLEAN=1 ;;
        --test) RUN_TEST=1 ;;
        --) shift; APP_ARGS=("$@"); break ;;
        *) echo "::error::Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT="run.sh"
[ "$RUN_TEST" -eq 1 ] && SCRIPT="test.sh"

if [ "$USE_DOCKER" -eq 0 ]; then
    if [ -f install_runtime.sh ]; then
        echo "::step::Installing runtime"
        bash install_runtime.sh
    fi
    echo "::step::Running mediapipe_hand_gesture_ubuntu_py natively"
    exec bash "$SCRIPT" "${APP_ARGS[@]}"
fi

if [ ! -f "$APP_DIR/Dockerfile" ]; then
    echo "::error::No Dockerfile found for mediapipe_hand_gesture_ubuntu_py. Re-run with --no-docker to run natively." >&2
    exit 1
fi

source ../_shared/scripts/qairt_utils.sh

HASH="$(printf '%s' "$APP_DIR" | sha1sum | cut -c1-12)"
IMAGE_TAG="aiha-run-$(basename "$APP_DIR")-$HASH"

if [ "$CLEAN" -eq 1 ]; then
    echo "::step::Cleaning prior docker image"
    docker rmi "$IMAGE_TAG" >/dev/null 2>&1 || true
    echo "::done::clean"
fi

LIBCDSPRPC_SRC=""
if [ -f "/usr/lib/aarch64-linux-gnu/libcdsprpc.so" ]; then
    LIBCDSPRPC_SRC="/usr/lib/aarch64-linux-gnu/libcdsprpc.so"
elif [ -f "/usr/lib/libcdsprpc.so" ]; then
    LIBCDSPRPC_SRC="/usr/lib/libcdsprpc.so"
else
    echo "::error::libcdsprpc.so not found in /usr/lib/aarch64-linux-gnu/ or /usr/lib/" >&2
    exit 1
fi

echo "::step::Building Docker image"
docker build --build-arg BUILD_TYPE=runtime -t "$IMAGE_TAG" .
echo "::done::Docker image"

# Forward every QAI_HUB_APPS_* variable the CLI injected into the container
device_env_args=()
for var in "${!QAI_HUB_APPS_@}"; do
    device_env_args+=(-e "$var")
done

echo "::step::Running mediapipe_hand_gesture_ubuntu_py in Docker"
docker run --rm --privileged \
    -v /usr/lib/:/opt/host/lib/:ro \
    -v "$LIBCDSPRPC_SRC:/usr/lib/libcdsprpc.so:ro" \
    -v /tmp/socket/cam_server:/tmp/socket/cam_server \
    -v "$QAIRT_ROOT:$QAIRT_ROOT" \
    -p 8080:8080 \
    "${device_env_args[@]}" \
    -w /app \
    "$IMAGE_TAG" bash "$SCRIPT" "${APP_ARGS[@]}"
echo "::done::run"
