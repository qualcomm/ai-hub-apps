# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Loads versions.env (KEY=VALUE) into PowerShell variables in the caller's scope.
#
# Usage: . load_versions.ps1
# ---------------------------------------------------------------------
$_LoadVersionsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$_VersionsFile = Join-Path $_LoadVersionsDir "versions.env"

if (-not (Test-Path $_VersionsFile)) {
    Write-Error "versions.env not found at $_VersionsFile"
    exit 1
}

Get-Content $_VersionsFile |
    Where-Object { $_ -match '=' -and $_ -notmatch '^\s*#' } |
    ConvertFrom-StringData |
    ForEach-Object { $_.GetEnumerator() } |
    ForEach-Object { Set-Variable -Name $_.Key -Value ($_.Value.Trim('"').Trim("'")) }
