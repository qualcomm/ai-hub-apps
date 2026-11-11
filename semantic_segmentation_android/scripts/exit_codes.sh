#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Exit codes an app's launch/run/test script uses to signal the CLI.
#
# Variables:
#   QAIHA_EXIT_BUILD_REQUIRED
#       The app has no build output yet. 'qai-hub-apps run' builds the app
#       and retries once; a second occurrence is a hard failure. Kept in sync
#       with BUILD_REQUIRED_EXIT_CODE in qai_hub_apps.experimental.commands.run
#       and with exit_codes.ps1.
#
# Usage: source exit_codes.sh
# ---------------------------------------------------------------------

# shellcheck disable=SC2034  # read by the sourcing script, not by this one.
QAIHA_EXIT_BUILD_REQUIRED=86
