#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail

# QC Linux devices boot with a read-only rootfs and run as root, so it must be
# remounted rw. Ubuntu devices are already writable and needs sudo, so the
# remount is skipped there. Anything unrecognized is treated as QC Linux.
is_ubuntu() {
    grep -qiE '^(ID|ID_LIKE)=.*(ubuntu|debian)' /etc/os-release 2>/dev/null
}

remount_rootfs_rw() {
    if is_ubuntu; then
        echo "Ubuntu device detected; skipping rootfs remount."
    else
        mount -o rw,remount /
    fi
}

TEST_CONTENT_DIR=/data/local/tmp/TestContent
APP_DIR="$TEST_CONTENT_DIR/app"
LOG_DIR=/data/local/tmp/QDC_logs
# set QAIHA_APP_ROOT for shared utils
export QAIHA_APP_ROOT="$APP_DIR"
USE_DOCKER="<<USE_DOCKER>>"

mkdir -p "$LOG_DIR"
# tee rather than redirect: a `set -e` abort (e.g. a failed assignment, which
# bash reports nothing for) otherwise leaves script.log empty and the job's
# console log silent, with no indication of where it stopped.
exec > "$LOG_DIR/script.log" 2>&1
trap 'echo "run_linux.sh: aborted at line $LINENO (exit $?)" >&2' ERR

remount_rootfs_rw

cd "$APP_DIR"

# Install the qai-hub-apps CLI, then run the app's on-device test through it. The
# CLI's launch.sh owns install_runtime and docker/native execution, so this script
# does not duplicate that logic.
export QAI_HUB_APPS_EXPERIMENTAL=1
export QAI_HUB_APPS_LOG_LEVEL=debug
# No TTY on the QDC device; skip install-time approval prompts.
export NON_INTERACTIVE=true

# The device Python has no venv module, so uv provisions an isolated Python and
# venv for the CLI. QC Linux ships pip as `pip3`; Ubuntu devices ship no pip at
# all, so pip itself has to be bootstrapped there before uv can be installed.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo -n"
fi

# Ubuntu 23.04+ marks the system Python as externally managed (PEP 668) and
# rejects a plain install; retry with the opt-out flag where it is supported.
pip_install_uv() {
    "$@" install uv || "$@" install --break-system-packages uv
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi

    for pip in pip3 pip; do
        if command -v "$pip" >/dev/null 2>&1; then
            pip_install_uv "$pip"
            return 0
        fi
    done

    # No pip on the device. ensurepip is stdlib and needs no network, but Debian
    # and Ubuntu strip it out of the base python3 package, so fall back to apt.
    if ! python3 -m ensurepip --upgrade; then
        $SUDO apt-get update
        $SUDO apt-get install -y python3-pip
    fi
    pip_install_uv python3 -m pip
}

install_uv
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
uv python install "<<PYTHON_VERSION>>"

CLI_VENV=/data/local/tmp/cli-venv
uv venv --python "<<PYTHON_VERSION>>" "$CLI_VENV"
# shellcheck disable=SC1091
source "$CLI_VENV/bin/activate"

# The CLI is a bundled wheel; its dependencies resolve from PyPI.
uv pip install --pre "<<CLI_SPEC>>"

REGISTRY_PATH="<<REGISTRY_PATH>>"
TEST_ARGS=(--app-path "$APP_DIR" --device "<<DEVICE_NAME>>" --model-id "<<MODEL_ID>>")
[ -n "$REGISTRY_PATH" ] && TEST_ARGS+=(--registry "$REGISTRY_PATH")
[ "$USE_DOCKER" = "false" ] && TEST_ARGS+=(--no-docker)

# QDC devices run the Adreno drivers, whose CDI and env files differ from the
# ones a user's device pins, so an IMSDK app fetches the QDC ones instead.
# Temporary: https://github.com/qcom-ai-hub/tetracode/issues/22430
IMSDK_DEVICE_ASSETS="$APP_DIR/scripts/assets/imsdk_utils/device_assets.json"
if [ -f "$IMSDK_DEVICE_ASSETS" ]; then
    cp "$TEST_CONTENT_DIR/device_assets.json" "$IMSDK_DEVICE_ASSETS"
    echo "Replaced $IMSDK_DEVICE_ASSETS with the QDC device assets."
fi

qai-hub-apps test "${TEST_ARGS[@]}"

remount_rootfs_rw

touch /data/local/tmp/QDCTestDone.txt
