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
#       --venv <dir>    : venv directory (default: $QAIHA_APP_ROOT/.venv, else $PWD/.venv)
#       --python <exe>  : Python executable to use for venv creation (default: python$PYTHON_VERSION)
#       <pkg_or_req>    : package spec (e.g. numpy==1.24) or -r requirements.txt
#       -- <args>       : extra flags passed directly to uv pip install
#
#   create_venv [--venv <dir>] [--python <exe>] [-- <venv_args>]
#       Create the venv if it does not exist yet, at the same directory
#       install_pip_deps and activate_venv resolve. Args after -- go to
#       `python -m venv` (e.g. --system-site-packages). To install into it
#       yourself, activate_venv and use `python -m pip`.
#
#   activate_venv [<dir>]
#       Activate the venv install_pip_deps created. Resolves the same directory,
#       so callers normally pass nothing.
#
# $QAIHA_VENV_OVERRIDE, when set, takes precedence over both the default and an
# explicit --venv/<dir>.
#
# Usage: source pip_utils.sh
# ---------------------------------------------------------------------
_PIP_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_PIP_UTILS_DIR/load_versions.sh"
# shellcheck disable=SC1091
source "$_PIP_UTILS_DIR/interactive.sh"

# Resolve the venv directory from (in order) $QAIHA_VENV_OVERRIDE, the caller's
# request, and the app root.
_resolve_venv_dir() {
    if [ -n "${QAIHA_VENV_OVERRIDE:-}" ]; then
        printf '%s' "$QAIHA_VENV_OVERRIDE"
    elif [ -n "${1:-}" ]; then
        printf '%s' "$1"
    else
        printf '%s' "${QAIHA_APP_ROOT:-$PWD}/.venv"
    fi
}

create_venv() {
    local venv_dir=""
    local python_exe=""
    local -a venv_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --venv)   venv_dir="$2"; shift 2 ;;
            --python) python_exe="$2"; shift 2 ;;
            --) shift; venv_args=("$@"); break ;;
            *) echo "error: create_venv: unknown argument '$1'" >&2; return 2 ;;
        esac
    done

    venv_dir="$(_resolve_venv_dir "$venv_dir")"
    if [ -x "$venv_dir/bin/python" ]; then
        return 0
    fi
    echo "::step::Creating virtual environment at $venv_dir"
    "${python_exe:-python${PYTHON_VERSION}}" -m venv "${venv_args[@]}" "$venv_dir"
    echo "::done::virtual environment"
}

activate_venv() {
    local venv_dir
    venv_dir="$(_resolve_venv_dir "${1:-}")"
    if [ ! -f "$venv_dir/bin/activate" ]; then
        echo "error: virtual environment not found at $venv_dir" >&2
        return 1
    fi
    # Dot rather than source: the shell bundler rewrites whole-line "source"
    # statements and warns on ones it cannot resolve to a shared script.
    # shellcheck disable=SC1091
    . "$venv_dir/bin/activate"
}

_install_pip_deps() {
    local venv_dir=""
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

    local -a create_args=()
    [ -n "$venv_dir" ] && create_args+=(--venv "$venv_dir")
    [ -n "$python_exe" ] && create_args+=(--python "$python_exe")
    create_venv "${create_args[@]}"
    venv_dir="$(_resolve_venv_dir "$venv_dir")"

    echo "::step::Installing uv"
    "$venv_dir/bin/python" -m pip install --quiet uv

    echo "::step::Installing Python dependencies"
    "$venv_dir/bin/uv" pip install --python "$venv_dir/bin/python" "${install_args[@]}" "${extra_args[@]}" --system-certs
    echo "::done::pip install"
}

install_pip_deps() {
    require_consent "Create/populate a virtual environment and install Python dependencies" \
        -- _install_pip_deps "$@"
}
