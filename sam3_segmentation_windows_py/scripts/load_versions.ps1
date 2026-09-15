# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Loads versions.env (KEY=VALUE) into PowerShell variables in the caller's scope.
# If $QAIHA_APP_ROOT\versions.override.env exists, its keys are layered on top
# of the global versions.env (the override wins).
#
# Usage: . load_versions.ps1
# ---------------------------------------------------------------------
$_LoadVersionsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$_VersionsFile = Join-Path $_LoadVersionsDir "versions.env"

if ([string]::IsNullOrEmpty($env:QAIHA_APP_ROOT)) {
    Write-Error ("QAIHA_APP_ROOT environment variable is required to use this utility. Set using '`$env:QAIHA_APP_ROOT = <app dir>'")
    exit 1
}
if (-not (Test-Path $_VersionsFile)) {
    Write-Error "versions.env not found at $_VersionsFile"
    exit 1
}

foreach ($_file in @($_VersionsFile, (Join-Path $env:QAIHA_APP_ROOT "versions.override.env"))) {
    if (-not (Test-Path $_file)) { continue }
    Get-Content $_file |
        Where-Object { $_ -match '=' -and $_ -notmatch '^\s*#' } |
        ConvertFrom-StringData |
        ForEach-Object { $_.GetEnumerator() } |
        ForEach-Object { Set-Variable -Name $_.Key -Value ($_.Value.Trim('"').Trim("'")) }
}
