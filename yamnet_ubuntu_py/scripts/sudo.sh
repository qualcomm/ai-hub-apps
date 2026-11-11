#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Sets SUDO="sudo" when running as a non-root user with sudo available,
# otherwise SUDO="". Docker containers typically run as root without sudo
# installed, so this lets the same scripts run in containers and on a host.
# Source this before any privileged commands.
#
# Usage: source sudo.sh
# ---------------------------------------------------------------------
export SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi
