#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Sets SUDO="" when already running as root or sudo is unavailable,
# otherwise SUDO="sudo". Source this before any privileged commands.
#
# Usage: source sudo.sh
# ---------------------------------------------------------------------
export SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi
