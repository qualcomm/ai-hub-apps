#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Create a Python virtual environment and install the qai-hub-apps CLI package,
# which includes the internal tooling (qai_hub_apps._internal).
#
# Usage:
#   bash tools/setup_env.sh [--venv <path>] [--python <exe>] [--recreate] [--with-cli] [--with-qdc-sdk]
#
# Defaults:
#   --venv    qaiha-dev
#   --python  python3
#
# --python takes an interpreter executable name or path (e.g. python3.12,
# /usr/bin/python3.12), not a bare version like 3.12.
#
# Flags:
#   --recreate      Delete and recreate the venv if it already exists
#   --with-cli      Also install the qai-hub-apps CLI package (cli/)
#   --with-qdc-sdk  Download and install the Qualcomm Device Cloud SDK wheel
#
# If the venv already exists with a different Python major.minor than --python,
# the script fails; pass --recreate or a different --venv.

set -euo pipefail

VENV_PATH="qaiha-dev"
PYTHON="python3"
RECREATE=false
WITH_CLI=false
WITH_QDC_SDK=false

while [ $# -gt 0 ]; do
    case $1 in
        --venv=*)        VENV_PATH="${1##--venv=}" ;;
        --venv)          VENV_PATH="$2"; shift ;;
        --python=*)      PYTHON="${1##--python=}" ;;
        --python)        PYTHON="$2"; shift ;;
        --recreate)      RECREATE=true ;;
        --with-cli)      WITH_CLI=true ;;
        --with-qdc-sdk)  WITH_QDC_SDK=true ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# shellcheck source=tools/ci/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ci/common.sh"
REPO_ROOT="$(repo_root)"

if ! command -v "$PYTHON" &>/dev/null; then
    echo "[ERROR] Python executable not found: $PYTHON (--python takes an executable, e.g. python3.12)" >&2
    exit 1
fi

py_version() {
    "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])'
}

VENV_PYTHON="$VENV_PATH/bin/python"

if [ -d "$VENV_PATH" ] && [ "$RECREATE" = false ]; then
    if [ ! -x "$VENV_PYTHON" ] || ! VENV_VERSION="$(py_version "$VENV_PYTHON" 2>/dev/null)"; then
        echo "[ERROR] $VENV_PATH exists but is not a working virtual environment." >&2
        echo "        Re-run with --recreate, or delete $VENV_PATH." >&2
        exit 1
    fi
    REQUESTED_VERSION="$(py_version "$PYTHON")" || { echo "[ERROR] Failed to run $PYTHON" >&2; exit 1; }
    if [ "$VENV_VERSION" != "$REQUESTED_VERSION" ]; then
        echo "[ERROR] $VENV_PATH uses Python $VENV_VERSION, but $PYTHON is Python $REQUESTED_VERSION." >&2
        echo "        Re-run with --recreate, pass a different --venv, or delete $VENV_PATH." >&2
        exit 1
    fi
    echo "Virtual environment already exists at $VENV_PATH (Python $VENV_VERSION)"
else
    if [ -d "$VENV_PATH" ]; then
        echo "Removing existing virtual environment at $VENV_PATH"
        rm -rf "$VENV_PATH"
    fi
    echo "Creating virtual environment at $VENV_PATH using $PYTHON"
    "$PYTHON" -m venv "$VENV_PATH"
fi

INSTALL_TARGET="$REPO_ROOT/cli/[dev]"

if command -v uv &>/dev/null; then
    uv pip install --python "$VENV_PYTHON" -e "$INSTALL_TARGET"
else
    "$VENV_PATH/bin/pip" install -e "$INSTALL_TARGET"
fi

if [ "$WITH_CLI" = true ]; then
    echo "Note: --with-cli is a no-op; the CLI package is installed above."
fi

# The Java/Kotlin pre-commit hooks need a JVM toolchain
# (JDK + google-java-format + ktlint).
bash "$REPO_ROOT/tools/lint/install_lint_tools.sh"

if [ "$WITH_QDC_SDK" = true ]; then
    echo "Downloading and installing QDC SDK wheel..."
    QDC_TMP_DIR="$(mktemp -d)"
    bash "$REPO_ROOT/tools/ci/download-qdc-wheel.sh" "$QDC_TMP_DIR"
    wheel=("$QDC_TMP_DIR"/*.whl)
    if command -v uv &>/dev/null; then
        uv pip install --python "$VENV_PYTHON" "${wheel[0]}"
    else
        "$VENV_PATH/bin/pip" install "${wheel[0]}"
    fi
    rm -rf "$QDC_TMP_DIR"
fi

echo ""
echo "Done. Activate with: source $VENV_PATH/bin/activate"
