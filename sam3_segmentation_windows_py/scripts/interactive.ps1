# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# User-consent gating for install/system-mutating commands.
#
# Wraps a script block and asks the user for y/N approval before running it.
# Approval is skipped when $env:NON_INTERACTIVE -eq "true".
#
# Functions:
#   Invoke-WithConsent -Description <str> [-SkipOnDecline] -Action { <block> }
#       Prompt the user to confirm <Description>, then run <Action>.
#       - NON_INTERACTIVE=true or INTERACTIVE_GRANTED=1 -> run without asking.
#       - yes -> run <Action> with $env:INTERACTIVE_GRANTED=1 set for its
#                duration, so nested Invoke-WithConsent calls auto-pass (one
#                approval covers everything the action runs internally).
#       - no  -> throw (abort) by default, or, with -SkipOnDecline, emit
#                ::skip:: and return so the caller can continue.
#
# Usage: . interactive.ps1
# ---------------------------------------------------------------------

function Invoke-WithConsent {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [switch]$SkipOnDecline,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    # Already approved (NON_INTERACTIVE or an enclosing consented action).
    if ($env:NON_INTERACTIVE -eq "true" -or $env:INTERACTIVE_GRANTED -eq "1") {
        & $Action
        return
    }

    $answer = Read-Host "$Description [y/N]"
    if ($answer -match '^(y|yes)$') {
        $prev = $env:INTERACTIVE_GRANTED
        $env:INTERACTIVE_GRANTED = "1"
        try {
            & $Action
        } finally {
            $env:INTERACTIVE_GRANTED = $prev
        }
    } else {
        if ($SkipOnDecline) {
            Write-Host "::skip::$Description (declined)"
            return
        }
        throw "Aborted: $Description (declined)"
    }
}
