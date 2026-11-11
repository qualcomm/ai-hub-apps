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

source ../_shared/scripts/runtime_utils.sh
STAMP_FILES=(install_runtime.sh requirements*.txt versions.override.env scripts/versions.env)

# The Adreno/FastRPC/QNN drivers live on the host, so they are installed here
# rather than in install_runtime.sh, which the docker path runs in the container.
source ../_shared/scripts/qcom_host_utils.sh
install_qcom_host_pkgs

if [ "$USE_DOCKER" -eq 0 ]; then
    if [ -f install_runtime.sh ]; then
        RUNTIME_MARKER="$APP_DIR/.qaiha_runtime_installed"
        if [ "$CLEAN" -eq 1 ] && [ -f "$RUNTIME_MARKER" ]; then
            echo "--clean: removing runtime marker at $RUNTIME_MARKER"
            rm -f "$RUNTIME_MARKER"
        fi
        install_runtime_native "$RUNTIME_MARKER" "$(runtime_stamp "${STAMP_FILES[@]}")"
    fi
    echo "::step::Running object_detection_3d_ubuntu_py natively"
    exec bash "$SCRIPT" "${APP_ARGS[@]}"
fi

if [ ! -f "$APP_DIR/Dockerfile" ]; then
    echo "::error::No Dockerfile found for object_detection_3d_ubuntu_py. Re-run with --no-docker to run natively." >&2
    exit 1
fi

source ../_shared/scripts/sudo.sh
source ../_shared/scripts/qairt_utils.sh
source ../_shared/scripts/docker_utils.sh

IMAGE_TAG="$(docker_image_tag aiha-run "$APP_DIR")"
CONTAINER_NAME="$IMAGE_TAG-container"

# The venv must live outside /app, or the bind mount would hide it -- and a
# container-built venv in the app dir would collide with the native one.
CONTAINER_VENV_DIR="/opt/qaiha/venv"
CONTAINER_RUNTIME_MARKER="/opt/qaiha/runtime-installed"

if [ "$CLEAN" -eq 1 ]; then
    echo "::step::Cleaning prior docker container and image"
    $SUDO docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    $SUDO docker rmi "$IMAGE_TAG" >/dev/null 2>&1 || true
    echo "::done::clean"
fi

# Bind-mounted into the container at the unversioned path the QAIRT HTP stub
# loads by name. install_qcom_host_pkgs above guarantees it exists.
LIBCDSPRPC_SRC="$(qcom_host_libcdsprpc)"

# The base image is prebuilt and published, so this build is just a pull.
BASE_IMAGE="$(resolve_base_image "ghcr.io/qcom-ai-hub/qai-hub-apps-ubuntu-base:sha-f3ce55a28d93")"

echo "::step::Building Docker image from $BASE_IMAGE"
if ! $SUDO docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$IMAGE_TAG" .; then
    echo "::error::Failed to build the image for object_detection_3d_ubuntu_py. If '$BASE_IMAGE' could not be pulled, check network access to it, or set QAIHA_BASE_IMAGE to a base image you built locally from tools/docker/ubuntu.dockerfile. Alternatively re-run with --no-docker to run natively." >&2
    exit 1
fi
echo "::done::Docker image"

# Environment evaluated per exec, never baked into the container: the
# QAI_HUB_APPS_* device variables change with --device, so a container created
# for one device would otherwise report that device forever.
exec_env_args=(-e "QAIHA_APP_ROOT=/app" -e "QAIHA_VENV_OVERRIDE=$CONTAINER_VENV_DIR")
for var in "${!QAI_HUB_APPS_@}"; do
    exec_env_args+=(-e "$var=${!var}")
done
# The container only sees its Docker bridge IP; pass the host's so the web UI
# can print a URL reachable from the network.
host_ip="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") {print $(i + 1); exit}}')"
if [ -n "$host_ip" ]; then
    exec_env_args+=(-e "QAIHA_HOST_IP=$host_ip")
fi

replace_stale_container "$CONTAINER_NAME" "$IMAGE_TAG"

# Reuse this app directory's container, or create it. The app directory is
# bind-mounted rather than copied into the image, so app edits cost nothing.
start_container "$CONTAINER_NAME" "$IMAGE_TAG" --privileged \
    -v "$APP_DIR:/app" \
    -v /usr/lib/:/opt/host/lib/:ro \
    -v "$LIBCDSPRPC_SRC:/usr/lib/libcdsprpc.so:ro" \
    -v /tmp/socket/cam_server:/tmp/socket/cam_server \
    -v "$QAIRT_ROOT:$QAIRT_ROOT" \
    -p 8080:8080

# Stop the container after every launch: docker-proxy holds port 8080 for as
# long as the container runs, not as long as the app runs, which would block the
# next app. Also hand back anything the container wrote into the bind-mounted
# app directory, which it wrote as root.
cleanup_container() {
    $SUDO docker exec "$CONTAINER_NAME" chown -R "$(id -u):$(id -g)" /app >/dev/null 2>&1 ||
        echo "::warning::Failed to reclaim ownership of $APP_DIR from $CONTAINER_NAME; files it wrote may still be owned by root." >&2
    $SUDO docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup_container EXIT

install_qc_ca_certs "$CONTAINER_NAME"

if [ -f install_runtime.sh ]; then
    install_runtime_in_container "$CONTAINER_NAME" "$CONTAINER_RUNTIME_MARKER" \
        "$(runtime_stamp "${STAMP_FILES[@]}")" "${exec_env_args[@]}"
fi

# -i so the app can read stdin, -t so Ctrl-C reaches it inside the container
# rather than only the local docker CLI. Only when stdin is a terminal, or
# docker exec refuses.
tty_args=()
if [ -t 0 ]; then
    tty_args=(-i -t)
fi

echo "::step::Running object_detection_3d_ubuntu_py in Docker"
$SUDO docker exec "${tty_args[@]}" "${exec_env_args[@]}" -w /app "$CONTAINER_NAME" \
    bash "$SCRIPT" "${APP_ARGS[@]}"
echo "::done::run"
