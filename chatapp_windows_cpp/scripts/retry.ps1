# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Retry wrapper for flaky commands.
#
# Re-runs a script block with exponential backoff until it succeeds or the
# attempt budget is exhausted.
#
# Functions:
#   Invoke-WithRetry -Description <str> [-Attempts N] [-Backoff BASE] -Action { <block> }
#       Run <Action>, retrying on failure.
#       -Attempts N  : total attempts (default 3).
#       -Backoff BASE: base backoff seconds (default 5). After a failed
#                      attempt, sleeps BASE * 2^(attempt-1): 5s, 10s, 15s, ...
#       A failure is a thrown exception or a non-zero $LASTEXITCODE from a
#       native command inside the block.
#       - success -> return immediately.
#       - all attempts fail -> re-throw the last error so callers abort.
#
# Usage: . retry.ps1
# ---------------------------------------------------------------------

function Invoke-WithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [int]$Attempts = 3,
        [int]$Backoff = 5,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        $failure = $null
        try {
            $global:LASTEXITCODE = 0
            & $Action
            if ($LASTEXITCODE -ne 0) {
                $failure = "exit $LASTEXITCODE"
            }
        } catch {
            $failure = $_
        }

        if (-not $failure) {
            return
        }

        if ($attempt -ge $Attempts) {
            Write-Host "::error::$Description failed after $Attempts attempt(s) ($failure)"
            throw "Aborted: $Description failed after $Attempts attempt(s) ($failure)"
        }

        $delay = $Backoff * [math]::Pow(2, $attempt - 1)
        Write-Host "::step::$Description failed ($failure); retrying in ${delay}s (attempt $($attempt + 1)/$Attempts)"
        Start-Sleep -Seconds $delay
    }
}
