#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Create a Python virtual environment and install qai_hub_apps_test.
#
# Usage:
#   bash tools/setup_env.sh [--venv <path>] [--python <exe>] [--extras <extra>] [--with-cli] [--with-qdc-sdk]
#
# Defaults:
#   --venv    qaiha-dev
#   --python  python3
#   --extras  dev
#
# Available extras:
#   dev        Full test install: pytest, qai_hub_models, boto3, etc. (default)
#   precommit  Light install: pre-commit + mypy only (for CI lint checks)
#
# Flags:
#   --with-cli      Also install the qai-hub-apps CLI package (cli/)
#   --with-qdc-sdk  Download and install the Qualcomm Device Cloud SDK wheel

set -euo pipefail

VENV_PATH="qaiha-dev"
PYTHON="python3"
EXTRAS="dev"
WITH_CLI=false
WITH_QDC_SDK=false

while [ $# -gt 0 ]; do
    case $1 in
        --venv=*)        VENV_PATH="${1##--venv=}" ;;
        --venv)          VENV_PATH="$2"; shift ;;
        --python=*)      PYTHON="${1##--python=}" ;;
        --python)        PYTHON="$2"; shift ;;
        --extras=*)      EXTRAS="${1##--extras=}" ;;
        --extras)        EXTRAS="$2"; shift ;;
        --with-cli)      WITH_CLI=true ;;
        --with-qdc-sdk)  WITH_QDC_SDK=true ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

REPO_ROOT="$(git rev-parse --show-toplevel)"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH using $PYTHON"
    "$PYTHON" -m venv "$VENV_PATH"
else
    echo "Virtual environment already exists at $VENV_PATH"
fi

INSTALL_TARGET="$REPO_ROOT/tools/python/[$EXTRAS]"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_VERSION="torch==2.8.0+cpu"

# Pre-install CPU torch before qai_hub_models so pip doesn't pull in the CUDA build.
if [ "$EXTRAS" = "dev" ]; then
    if command -v uv &>/dev/null; then
        uv pip install --python "$VENV_PATH/bin/python" --extra-index-url "$TORCH_INDEX" "$TORCH_VERSION"
    else
        "$VENV_PATH/bin/pip" install --extra-index-url "$TORCH_INDEX" "$TORCH_VERSION"
    fi
fi

if command -v uv &>/dev/null; then
    uv pip install --python "$VENV_PATH/bin/python" -e "$INSTALL_TARGET"
else
    "$VENV_PATH/bin/pip" install -e "$INSTALL_TARGET"
fi

if [ "$WITH_CLI" = true ]; then
    echo "Installing CLI package (cli/)..."
    if command -v uv &>/dev/null; then
        uv pip install --python "$VENV_PATH/bin/python" -e "$REPO_ROOT/cli/"
    else
        "$VENV_PATH/bin/pip" install -e "$REPO_ROOT/cli/"
    fi
fi

if [ "$WITH_QDC_SDK" = true ]; then
    echo "Downloading and installing QDC SDK wheel..."
    QDC_TMP_DIR="$(mktemp -d)"
    bash "$REPO_ROOT/tools/ci/download-qdc-wheel.sh" "$QDC_TMP_DIR"
    wheel=("$QDC_TMP_DIR"/*.whl)
    if command -v uv &>/dev/null; then
        uv pip install --python "$VENV_PATH/bin/python" "${wheel[0]}"
    else
        "$VENV_PATH/bin/pip" install "${wheel[0]}"
    fi
    rm -rf "$QDC_TMP_DIR"
fi

echo ""
echo "Done. Activate with: source $VENV_PATH/bin/activate"
