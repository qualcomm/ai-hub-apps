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
for arg in "$@"; do
    case "$arg" in
        --no-docker) USE_DOCKER=0 ;;
        --docker) USE_DOCKER=1 ;;
        --clean) CLEAN=1 ;;
        *) echo "::error::Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ "$USE_DOCKER" -eq 0 ]; then
    echo "::error::Android apps require Docker to build (no native build)." >&2
    exit 1
fi

if [ ! -f "$APP_DIR/Dockerfile" ]; then
    echo "::error::No Dockerfile found for super_resolution_android; it cannot be built." >&2
    exit 1
fi

# The image bakes the Android toolchain from the bundled shared scripts, which
# only exist in a fetched/bundled app. Fail with a clear message rather than a COPY error.
if [ ! -d "$APP_DIR/scripts" ]; then
    echo "::error::No scripts/ directory found for super_resolution_android. Docker builds need a bundled app; use 'qai-hub-apps fetch super_resolution_android'." >&2
    exit 1
fi

# Derive unique image/container names from the app directory so two copies of
# the same app in different directories never collide.
HASH="$(printf '%s' "$APP_DIR" | sha1sum | cut -c1-12)"
IMAGE_TAG="aiha-build-$(basename "$APP_DIR")-$HASH"
CONTAINER_NAME="$IMAGE_TAG-container"

build_args=()
# The internal registry mirror and Qualcomm CA certs are only reachable from the
# Qualcomm internal network (CI runners or a corp-network machine). Set
# QC_INTERNAL_HOST=1 there. Otherwise the Dockerfile defaults apply: the public
# base image and no CA injection.
if [ "${QC_INTERNAL_HOST:-}" = "1" ]; then
    build_args+=(
        --build-arg REGISTRY_PREFIX=docker-registry.qualcomm.com/library/
        --build-arg INSTALL_QUALCOMM_CA=true
    )
fi
# --clean tears down prior build state (image, container, host-side outputs) and
# rebuilds the image from scratch.
if [ "$CLEAN" -eq 1 ]; then
    echo "::step::Cleaning prior build outputs, docker image and container"
    build_args+=(--no-cache)
    rm -rf ./build/outputs
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rmi "$IMAGE_TAG" >/dev/null 2>&1 || true
    echo "::done::clean"
fi

echo "::step::Building Docker image"
docker build "${build_args[@]}" -t "$IMAGE_TAG" .
echo "::done::Docker image"

# Reuse this app directory's container, or create it.
if ! docker start "$CONTAINER_NAME" >/dev/null 2>&1; then
    # A create or install that died earlier can leave a container behind under
    # this name in a state docker start rejects; replace it.
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "::step::Creating container $CONTAINER_NAME"
    # --init reaps the daemons that docker exec children reparent to PID 1.
    docker create --name "$CONTAINER_NAME" --init \
        -v "$APP_DIR:/app" \
        "$IMAGE_TAG" sleep infinity >/dev/null
    docker start "$CONTAINER_NAME" >/dev/null
    echo "::done::container"
fi

# Stop the container after the build so it holds no resources between builds,
# and hand back anything it wrote into the bind-mounted app directory, which it
# wrote as root -- otherwise --clean's "rm -rf ./build/outputs" and a re-fetch
# both fail with EPERM.
cleanup_container() {
    docker exec "$CONTAINER_NAME" chown -R "$(id -u):$(id -g)" /app >/dev/null 2>&1 || true
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup_container EXIT

if [ -f install_build.sh ]; then
    echo "::step::Installing build dependencies in $CONTAINER_NAME"
    docker exec -w /app "$CONTAINER_NAME" bash install_build.sh
    echo "::done::Installing build dependencies"
fi

echo "::step::Building APKs (gradle assembleDebug assembleAndroidTest)"
docker exec -w /app "$CONTAINER_NAME" bash -c '
    set -euo pipefail
    . /app/scripts/android_utils.sh
    if [ -f /app/scripts/qairt_utils.sh ]; then
        . /app/scripts/qairt_utils.sh
    fi
    gradle assembleDebug assembleAndroidTest'
echo "::done::APKs built into $APP_DIR/build/outputs"
