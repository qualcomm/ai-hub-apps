#!/bin/sh
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# shellcheck disable=SC2086
IMAGE="aiha-hand-gesture"
DOCKER_OPTS="--rm --privileged \
    -v /usr/lib/:/opt/host/lib/:ro \
    -v /usr/lib/aarch64-linux-gnu/libcdsprpc.so:/usr/lib/libcdsprpc.so:ro \
    -v $(pwd):/root \
    -p 8080:8080"

if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
    sudo docker run $DOCKER_OPTS --entrypoint '' -it $IMAGE bash
else
    sudo docker run $DOCKER_OPTS $IMAGE "$@"
fi
