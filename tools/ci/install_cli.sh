# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# Installs the qai-hub-apps CLI from one of several sources:
#   source  -> editable install from the checked-out repo (cli/); no version needed
#   s3      -> published wheel from the self-hosted dev wheel index
#   staging -> published wheel from test.pypi.org
#   prod    -> published wheel from pypi.org
#
# Published installs pin the version (--version), pass --pre (no-op for finals,
# required for dev/rc pins), resolve transitive deps from real PyPI, and retry.
#
# With --venv, installs into that venv's python; otherwise
# installs into the active environment.
#
# Usage:
#   install_cli.sh --source source [--venv <path>]
#   install_cli.sh --source {s3|staging|prod} --version <version> [--venv <path>]
# ---------------------------------------------------------------------

set -euo pipefail

# shellcheck source=tools/ci/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SOURCE=""
VERSION=""
VENV=""

while [ $# -gt 0 ]; do
    case $1 in
        --source) SOURCE="$2"; shift ;;
        --source=*) SOURCE="${1#--source=}" ;;
        --version) VERSION="$2"; shift ;;
        --version=*) VERSION="${1#--version=}" ;;
        --venv) VENV="$2"; shift ;;
        --venv=*) VENV="${1#--venv=}" ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# Tool-specific extra args (e.g. trusting a plain-HTTP index); set per source below.
PIP_EXTRA_ARGS=()
UV_EXTRA_ARGS=()

# Install into a specific venv's python when --venv is given, else the active env.
pip_install() {
    UV_AVAILABLE=$(command -v uv >/dev/null 2>&1 && echo true || echo false)
    if [ -n "$VENV" ]; then
        if [ "$UV_AVAILABLE" = true ]; then
            uv pip install --no-cache --python "$VENV/bin/python" "${UV_EXTRA_ARGS[@]}" "$@"
        else
            "$VENV/bin/pip" install --no-cache-dir "${PIP_EXTRA_ARGS[@]}" "$@"
        fi
    else
        if [ "$UV_AVAILABLE" = true ]; then
            uv pip install --no-cache "${UV_EXTRA_ARGS[@]}" "$@"
        else
            pip install --no-cache-dir "${PIP_EXTRA_ARGS[@]}" "$@"
        fi
    fi
}

if [ "$SOURCE" = "source" ]; then
    echo "Installing qai-hub-apps (editable) from cli/"
    pip_install -e "$(repo_root)/cli/"
    exit 0
fi

case "$SOURCE" in
    s3)
        HOST="qaihub-public-python-wheels.s3-website-us-west-2.amazonaws.com"
        INDEX_URL="http://$HOST/"
        # pip/uv ignore a plain-HTTP index unless the host is explicitly trusted.
        PIP_EXTRA_ARGS=(--trusted-host "$HOST")
        # --index-strategy: uv otherwise locks qai-hub-apps to the pypi --extra-index-url.
        UV_EXTRA_ARGS=(--allow-insecure-host "$HOST" --index-strategy unsafe-best-match)
        ;;
    staging)
        INDEX_URL="https://test.pypi.org/simple/"
        UV_EXTRA_ARGS=(--index-strategy unsafe-best-match)
        ;;
    prod)    INDEX_URL="https://pypi.org/simple/" ;;
    *) echo "ERROR: --source must be 'source', 's3', 'staging', or 'prod', got '$SOURCE'"; exit 1 ;;
esac

[ -n "$VERSION" ] || { echo "ERROR: --version is required for source '$SOURCE'"; exit 1; }

V="${VERSION#v}"
echo "Installing qai-hub-apps==$V from $SOURCE ($INDEX_URL)"
for i in {1..10}; do
    pip_install \
        --pre \
        --index-url "$INDEX_URL" \
        --extra-index-url "https://pypi.org/simple/" \
        "qai-hub-apps==$V" && break
    echo "Attempt $i failed, retrying in 60s..."
    sleep 60
    if [ "$i" -eq 10 ]; then
        echo "ERROR: Failed to install qai-hub-apps==$V after 10 attempts"
        exit 1
    fi
done
