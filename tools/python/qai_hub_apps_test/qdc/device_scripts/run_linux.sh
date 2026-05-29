#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

set -euo pipefail

mount -o rw,remount /

APP_DIR=/data/local/tmp/TestContent/app
LOG_DIR=/data/local/tmp/QDC_logs
# shellcheck disable=SC1072,SC1073
USE_DOCKER=<<USE_DOCKER>>

mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/script.log" 2>&1

cd "$APP_DIR"

if [ "$USE_DOCKER" = "true" ]; then
    # shellcheck source=/dev/null
    source "$APP_DIR/scripts/qairt_utils.sh"

    IMAGE_NAME="aiha-app-test"

    echo "Building Docker image ..."
    docker build \
        --build-arg BUILD_TYPE=runtime \
        -t "$IMAGE_NAME" "$APP_DIR"

    echo "Running inside container ..."
    docker run --rm --privileged \
        -v "$QAIRT_ROOT:$QAIRT_ROOT" \
        -v /usr/lib/libcdsprpc.so:/usr/lib/libcdsprpc.so:ro \
        -v /usr/lib/libcdsprpc.so.1:/usr/lib/libcdsprpc.so.1:ro \
        -v /usr/lib/libcdsprpc.so.1.0.0:/usr/lib/libcdsprpc.so.1.0.0:ro \
        -v /usr/lib/libdmabufheap.so.0:/usr/lib/libdmabufheap.so.0:ro \
        -v /usr/lib/libdmabufheap.so.0.0.0:/usr/lib/libdmabufheap.so.0.0.0:ro \
        -w /app \
        "$IMAGE_NAME" \
        bash -euo pipefail -c '<<RUN_COMMAND>>'
else
    if [ -f "install_runtime.sh" ]; then
        echo "Running install_runtime.sh ..."
        bash install_runtime.sh
    fi
    echo "Running app command ..."
    <<RUN_COMMAND>>
fi

mount -o rw,remount /

touch /data/local/tmp/QDCTestDone.txt
