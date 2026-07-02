#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# shellcheck disable=SC2086
IMAGE="aiha-gesture"
source "$(dirname "${BASH_SOURCE[0]}")/scripts/qairt_utils.sh"

LIBCDSPRPC_SRC=""
if [ -f "/usr/lib/aarch64-linux-gnu/libcdsprpc.so" ]; then
    LIBCDSPRPC_SRC="/usr/lib/aarch64-linux-gnu/libcdsprpc.so"
elif [ -f "/usr/lib/libcdsprpc.so" ]; then
    LIBCDSPRPC_SRC="/usr/lib/libcdsprpc.so"
else
    echo "Error: libcdsprpc.so not found in /usr/lib/aarch64-linux-gnu/ or /usr/lib/" >&2
    exit 1
fi

DOCKER_OPTS="--rm --privileged \
    -v /usr/lib/:/opt/host/lib/:ro \
    -v $LIBCDSPRPC_SRC:/usr/lib/libcdsprpc.so:ro \
    -v /tmp/socket/cam_server:/tmp/socket/cam_server \
    -v $QAIRT_ROOT:$QAIRT_ROOT \
    -p 8080:8080"
if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
    sudo docker run $DOCKER_OPTS -it $IMAGE bash
else
    sudo docker run $DOCKER_OPTS $IMAGE python main.py --qairt-path $QAIRT_PATH "$@"
fi
