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
    echo "::error::No Dockerfile found for geniex_chat_android; it cannot be built." >&2
    exit 1
fi

# install_build.sh and the gradle step source /app/scripts/*, which only exists in
# a fetched/bundled app.
if [ ! -d "$APP_DIR/scripts" ]; then
    echo "::error::No scripts/ directory found for geniex_chat_android. Docker builds need a bundled app; use 'qai-hub-apps fetch geniex_chat_android'." >&2
    exit 1
fi

source ../_shared/scripts/docker_utils.sh

IMAGE_TAG="$(docker_image_tag aiha-build "$APP_DIR")"
CONTAINER_NAME="$IMAGE_TAG-container"

# The Android toolchain lives in a prebuilt base image, so this build is just a
# pull.
BASE_IMAGE="$(resolve_base_image "ghcr.io/qcom-ai-hub/qai-hub-apps-android-base:sha-f3ce55a28d93")"
build_args=(--build-arg "BASE_IMAGE=$BASE_IMAGE")

# --clean tears down prior build state (image, container, host-side outputs).
# --no-cache only invalidates this app's trivial FROM layer, not the base image;
# to force a fresh base, 'docker rmi' it or bump the tag in versions.env.
if [ "$CLEAN" -eq 1 ]; then
    echo "::step::Cleaning prior build outputs, docker image and container"
    build_args+=(--no-cache)
    rm -rf ./build/outputs
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rmi "$IMAGE_TAG" >/dev/null 2>&1 || true
    echo "::done::clean"
fi

echo "::step::Building Docker image from $BASE_IMAGE"
if ! docker build "${build_args[@]}" -t "$IMAGE_TAG" .; then
    echo "::error::Failed to build the image for geniex_chat_android. If '$BASE_IMAGE' could not be pulled, check network access to it, or set QAIHA_BASE_IMAGE to a base image you built locally." >&2
    exit 1
fi
echo "::done::Docker image"

# Reuse this app directory's container, or create it.
start_container "$CONTAINER_NAME" "$IMAGE_TAG" -v "$APP_DIR:/app"

# Stop the container after the build so it holds no resources between builds,
# and hand back anything it wrote into the bind-mounted app directory, which it
# wrote as root -- otherwise --clean's "rm -rf ./build/outputs" and a re-fetch
# both fail with EPERM.
cleanup_container() {
    docker exec "$CONTAINER_NAME" chown -R "$(id -u):$(id -g)" /app >/dev/null 2>&1 || true
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup_container EXIT

install_qc_ca_certs "$CONTAINER_NAME" --java-keystore

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
