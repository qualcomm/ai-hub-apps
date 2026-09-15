#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# shellcheck disable=SC2086
IMAGE="aiha-face-recognition"
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
    # Resolve --gallery-dir / --image / --output to absolute host paths and
    # bind-mount the directories that contain them into the container at the
    # same path, so the app can read the gallery and image (read-only) and write
    # the annotated output. Without this, user-supplied host paths are not
    # visible inside the container.
    ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --gallery-dir)
                host_path="$(realpath "$2")"
                DOCKER_OPTS="$DOCKER_OPTS -v $host_path:$host_path:ro"
                ARGS+=("$1" "$host_path")
                shift 2
                ;;
            --image)
                host_path="$(realpath "$2")"
                host_dir="$(dirname "$host_path")"
                DOCKER_OPTS="$DOCKER_OPTS -v $host_dir:$host_dir:ro"
                ARGS+=("$1" "$host_path")
                shift 2
                ;;
            --output)
                host_path="$(realpath -m "$2")"
                host_dir="$(dirname "$host_path")"
                DOCKER_OPTS="$DOCKER_OPTS -v $host_dir:$host_dir"
                ARGS+=("$1" "$host_path")
                shift 2
                ;;
            *)
                ARGS+=("$1")
                shift
                ;;
        esac
    done
    sudo docker run $DOCKER_OPTS $IMAGE bash -c \
        'source .venv/bin/activate && exec python main.py --qairt-path "$0" "$@"' \
        "$QAIRT_PATH" "${ARGS[@]}"
fi
