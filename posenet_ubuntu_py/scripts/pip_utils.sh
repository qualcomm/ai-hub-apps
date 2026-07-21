#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Ubuntu/Linux pip/venv installation utilities.
#
# Functions:
#   install_pip_deps [--venv <dir>] [--python <exe>] <pkg_or_req> [<pkg_or_req> ...] [-- <extra_uv_args>]
#       Create a .venv (if needed) and install packages or requirements files via uv.
#       --venv <dir>    : venv directory (default: $PWD/.venv)
#       --python <exe>  : Python executable to use for venv creation (default: python$PYTHON_VERSION)
#       <pkg_or_req>    : package spec (e.g. numpy==1.24) or -r requirements.txt
#       -- <args>       : extra flags passed directly to uv pip install
#
# Usage: source pip_utils.sh
# ---------------------------------------------------------------------
_PIP_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_PIP_UTILS_DIR/load_versions.sh"

install_pip_deps() {
    local venv_dir="${PWD}/.venv"
    local python_exe=""
    local -a install_args=()
    local -a extra_args=()
    local after_sep=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --venv)   venv_dir="$2"; shift 2 ;;
            --python) python_exe="$2"; shift 2 ;;
            --) after_sep=1; shift ;;
            *) if [[ $after_sep -eq 1 ]]; then
                   extra_args+=("$1")
               else
                   install_args+=("$1")
               fi
               shift ;;
        esac
    done

    local python_bin="${python_exe:-python${PYTHON_VERSION}}"

    if [ ! -x "$venv_dir/bin/python" ]; then
        echo "::step::Creating virtual environment at $venv_dir"
        "$python_bin" -m venv "$venv_dir"
        echo "::step::Installing uv"
        "$venv_dir/bin/python" -m pip install --quiet uv
        echo "::done::virtual environment"
    fi

    echo "::step::Installing Python dependencies"
    "$venv_dir/bin/uv" pip install --python "$venv_dir/bin/python" "${install_args[@]}" "${extra_args[@]}" --system-certs
    echo "::done::pip install"
}
