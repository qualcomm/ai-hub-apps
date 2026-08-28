#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail

mount -o rw,remount /

APP_DIR=/data/local/tmp/TestContent/app
LOG_DIR=/data/local/tmp/QDC_logs
# set QAIHA_APP_ROOT for shared utils
export QAIHA_APP_ROOT="$APP_DIR"
USE_DOCKER="<<USE_DOCKER>>"

mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/script.log" 2>&1

cd "$APP_DIR"

# Install the qai-hub-apps CLI, then run the app's on-device test through it. The
# CLI's launch.sh owns install_runtime and docker/native execution, so this script
# does not duplicate that logic.
export QAI_HUB_APPS_EXPERIMENTAL=1
export QAI_HUB_APPS_LOG_LEVEL=debug
# No TTY on the QDC device; skip install-time approval prompts.
export NON_INTERACTIVE=true

# The device Python has no venv module and apt is unavailable; use uv to provision
# an isolated Python and venv for the CLI.
pip3 install uv
export PATH="/root/.local/bin:$PATH"
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

qai-hub-apps test "${TEST_ARGS[@]}"

mount -o rw,remount /

touch /data/local/tmp/QDCTestDone.txt
