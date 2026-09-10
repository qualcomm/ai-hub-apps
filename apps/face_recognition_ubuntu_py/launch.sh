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
        echo "::done::Installing runtime"
    fi
    echo "::step::Running face_recognition_ubuntu_py natively"
    exec bash "$SCRIPT" "${APP_ARGS[@]}"
fi

if [ ! -f "$APP_DIR/Dockerfile" ]; then
    echo "::error::No Dockerfile found for face_recognition_ubuntu_py. Re-run with --no-docker to run natively." >&2
    exit 1
fi

source ../_shared/scripts/qairt_utils.sh

HASH="$(printf '%s' "$APP_DIR" | sha1sum | cut -c1-12)"
IMAGE_TAG="aiha-run-$(basename "$APP_DIR")-$HASH"
CONTAINER_NAME="$IMAGE_TAG-container"

# The venv must live outside /app, or the bind mount would hide it -- and a
# container-built venv in the app dir would collide with the native one.
CONTAINER_VENV_DIR="/opt/qaiha/venv"

if [ "$CLEAN" -eq 1 ]; then
    echo "::step::Cleaning prior docker container and image"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
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
docker build -t "$IMAGE_TAG" .
echo "::done::Docker image"

# Environment evaluated per exec, never baked into the container: the
# QAI_HUB_APPS_* device variables change with --device, so a container created
# for one device would otherwise report that device forever.
exec_env_args=(-e "QAIHA_APP_ROOT=/app" -e "QAIHA_VENV_OVERRIDE=$CONTAINER_VENV_DIR")
for var in "${!QAI_HUB_APPS_@}"; do
    exec_env_args+=(-e "$var=${!var}")
done

# A container is pinned to the image id it was created from, not to the tag, so
# an existing container built from an older image would silently keep running
# the stale one. Drop it and let the block below recreate it.
built_image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE_TAG")"
container_image_id="$(docker container inspect -f '{{.Image}}' "$CONTAINER_NAME" 2>/dev/null || true)"
if [ -n "$container_image_id" ] && [ "$container_image_id" != "$built_image_id" ]; then
    echo "::step::Replacing container $CONTAINER_NAME built from a stale image"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "::done::replace"
fi

# Reuse this app directory's container, or create it. The app directory is
# bind-mounted rather than copied into the image, so app edits cost nothing.
if ! docker start "$CONTAINER_NAME" >/dev/null 2>&1; then
    # A create or install that died earlier can leave a container behind under
    # this name in a state docker start rejects; replace it.
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "::step::Creating container $CONTAINER_NAME"
    # --init reaps the daemons that docker exec children reparent to PID 1.
    docker create --name "$CONTAINER_NAME" --init --privileged \
        -v "$APP_DIR:/app" \
        -v /usr/lib/:/opt/host/lib/:ro \
        -v "$LIBCDSPRPC_SRC:/usr/lib/libcdsprpc.so:ro" \
        -v /tmp/socket/cam_server:/tmp/socket/cam_server \
        -v "$QAIRT_ROOT:$QAIRT_ROOT" \
        -p 8080:8080 \
        "$IMAGE_TAG" sleep infinity >/dev/null
    docker start "$CONTAINER_NAME" >/dev/null
    echo "::done::container"
fi

# Stop the container after every launch: docker-proxy holds port 8080 for as
# long as the container runs, not as long as the app runs, which would block the
# next app. Also hand back anything the container wrote into the bind-mounted
# app directory, which it wrote as root.
cleanup_container() {
    docker exec "$CONTAINER_NAME" chown -R "$(id -u):$(id -g)" /app >/dev/null 2>&1 ||
        echo "::warning::Failed to reclaim ownership of $APP_DIR from $CONTAINER_NAME; files it wrote may still be owned by root." >&2
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup_container EXIT

# Runs on every launch and is expected to skip whatever is already installed.
# The container persists between launches, so what it installed is still there.
if [ -f install_runtime.sh ]; then
    echo "::step::Installing runtime in $CONTAINER_NAME"
    docker exec "${exec_env_args[@]}" -w /app \
        "$CONTAINER_NAME" bash install_runtime.sh
    echo "::done::Installing runtime"
fi

# -t so Ctrl-C reaches the app inside the container rather than only the local
# docker CLI. Only when stdin is a terminal, or docker exec refuses.
tty_args=()
if [ -t 0 ]; then
    tty_args=(-t)
fi

echo "::step::Running face_recognition_ubuntu_py in Docker"
docker exec "${tty_args[@]}" "${exec_env_args[@]}" -w /app "$CONTAINER_NAME" \
    bash "$SCRIPT" "${APP_ARGS[@]}"
echo "::done::run"
