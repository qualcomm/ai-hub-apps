# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Windows winget package installation utilities.
#
# Functions:
#   Install-WingetPackage -Id <package_id> [-ExtraArgs <string[]>]
#       Install a winget package if it is not already installed.
#   Install-WingetPackages -Ids <string[]>
#       Install multiple winget packages, each idempotently.
#
# Usage: . winget_utils.ps1
# ---------------------------------------------------------------------
$_WingetUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_WingetUtilsDir\load_versions.ps1"

# add winget to PATH if installed but not yet on PATH
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Error "winget not found. Please install it from https://learn.microsoft.com/en-us/windows/package-manager/winget/ and re-run."
    exit 1
}

function Install-WingetPackage {
    param(
        [string]$Id,
        [string[]]$ExtraArgs = @()
    )
    $list = winget list --id $Id --exact --accept-source-agreements 2>&1
    if ($LASTEXITCODE -eq 0 -and ($list -match [regex]::Escape($Id))) {
        Write-Host "::skip::$Id"
    } else {
        Write-Host "::step::Installing $Id"
        winget install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements @ExtraArgs
        Write-Host "::done::$Id"
    }
}

function Install-WingetPackages {
    param(
        [string[]]$Ids
    )
    foreach ($id in $Ids) {
        Install-WingetPackage -Id $id
    }
}
