#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# User-consent gating for install/system-mutating commands.
#
# Wraps an arbitrary command/function and asks the user for y/N approval
# before running it. Approval is skipped when NON_INTERACTIVE=true.
#
# Functions:
#   require_consent [--skip-on-decline] <description> -- <command> [args...]
#       Prompt the user to confirm <description>, then run <command>.
#       - NON_INTERACTIVE=true or INTERACTIVE_GRANTED=1 -> run without asking.
#       - yes -> run <command> with INTERACTIVE_GRANTED=1 exported for that
#                command's subtree, so nested require_consent calls auto-pass
#                (one approval covers everything the command runs internally).
#       - no  -> abort the script (exit 1) by default, or, with
#                --skip-on-decline, emit ::skip:: and return non-zero so the
#                caller can continue.
#
# Usage: source interactive.sh
# ---------------------------------------------------------------------

require_consent() {
    local skip_on_decline=0
    if [ "${1:-}" = "--skip-on-decline" ]; then
        skip_on_decline=1
        shift
    fi

    local description="$1"; shift
    if [ "${1:-}" = "--" ]; then
        shift
    fi

    # Already approved (NON_INTERACTIVE or an enclosing consented command).
    if [ "${NON_INTERACTIVE:-}" = "true" ] || [ "${INTERACTIVE_GRANTED:-}" = "1" ]; then
        "$@"
        return $?
    fi

    local answer=""
    printf '%s [y/N] ' "$description" >&2
    read -r answer || true

    case "$answer" in
        [yY] | [yY][eE][sS])
            INTERACTIVE_GRANTED=1 "$@"
            return $?
            ;;
        *)
            if [ "$skip_on_decline" -eq 1 ]; then
                echo "::skip::${description} (declined)"
                return 1
            fi
            echo "Aborted: ${description} (declined)" >&2
            exit 1
            ;;
    esac
}
