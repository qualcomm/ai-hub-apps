#!/usr/bin/env bash

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

set -euo pipefail
source ../_shared/scripts/interactive.sh

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
    echo "::error::No Dockerfile found for geniex_chat_android; it cannot be built." >&2
    exit 1
fi

# Derive unique image/container names from the app directory so two copies of
# the same app in different directories never collide.
HASH="$(printf '%s' "$APP_DIR" | sha1sum | cut -c1-12)"
IMAGE_TAG="aiha-build-$(basename "$APP_DIR")-$HASH"
CONTAINER_NAME="$IMAGE_TAG-container"

# The container is transient; remove it on exit. The image is kept for cache
# reuse (removed only by --clean).
trap 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true' EXIT

build_args=(--build-arg BUILD_TYPE=build)
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

echo "::step::Building APKs (gradle assembleDebug assembleAndroidTest)"
# A container from a prior run may still hold this name (e.g. after a hard kill
# that skipped the cleanup trap).
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    require_consent "A container named '$CONTAINER_NAME' already exists (likely a leftover from a previous run). Remove it?" \
        -- docker rm -f "$CONTAINER_NAME"
fi
docker run --name "$CONTAINER_NAME" "$IMAGE_TAG" \
    bash -c "source /app/scripts/android_utils.sh && cd /app && gradle assembleDebug assembleAndroidTest"

# Create ./build first so docker cp lands outputs at ./build/outputs, not ./build.
mkdir -p ./build
docker cp "$CONTAINER_NAME:/app/build/outputs" ./build
echo "::done::APKs built into $APP_DIR/build/outputs"
