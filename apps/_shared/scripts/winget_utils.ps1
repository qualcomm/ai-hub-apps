# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Windows winget package installation utilities.
#
# Functions:
#   Install-WingetPackage -Id <package_id> [-ExtraArgs <string[]>]
#       Install a winget package.
#   Install-WingetPackages -Ids <string[]>
#       Install multiple winget packages.
#
# Usage: . winget_utils.ps1
# ---------------------------------------------------------------------
$_WingetUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_WingetUtilsDir\load_versions.ps1"
. "$_WingetUtilsDir\interactive.ps1"
. "$_WingetUtilsDir\retry.ps1"

# add winget to PATH if installed but not yet on PATH
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Error "winget not found. Please install it from https://learn.microsoft.com/en-us/windows/package-manager/winget/ and re-run."
    exit 1
}

# winget adds the install dir to the machine PATH, but the running process won't see it; so
# temporarily reload PATH to resolve the exe, then restore the original PATH.
function Resolve-InstalledExe {
    param([string]$Name)
    $originalPath = $env:PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $exe = (Get-Command $Name -ErrorAction SilentlyContinue).Source
    $env:PATH = $originalPath
    return $exe
}

# No pre-check for an existing install. `winget list --id` correlates installed Add/Remove
# Programs entries to catalog packages heuristically, so it reports unrelated installs
# under the queried id (with Python 3.10 installed, Python.Python.3.12 matches). winget
# signals the mismatch in the version column, but only in package-specific ways not worth
# encoding here, and it is unfixed upstream (microsoft/winget-cli#6475, #6132). `winget
# install` makes the same decision itself, so let it own it.
function _Install-WingetPackage {
    param(
        [string]$Id,
        [string[]]$ExtraArgs = @()
    )
    # An already-installed package exits non-zero: winget converts the install to an
    # upgrade and finds nothing newer. Not a failure, so do not let it trip the retry.
    $alreadyInstalled = @(
        0x8A15002B, # UPDATE_NOT_APPLICABLE
        0x8A150061, # PACKAGE_ALREADY_INSTALLED
        0x8A15010D  # INSTALL_ALREADY_INSTALLED
    )
    Write-Host "::step::Installing $Id"
    Invoke-WithRetry -Description "winget install $Id" -Action {
        winget install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements @ExtraArgs
        # winget returns HRESULTs, which surface as negative ints; mask before comparing.
        if ($alreadyInstalled -contains ($LASTEXITCODE -band 0xFFFFFFFF)) {
            $global:LASTEXITCODE = 0
        }
    }
    Write-Host "::done::$Id"
}

function Install-WingetPackage {
    param(
        [string]$Id,
        [string[]]$ExtraArgs = @()
    )
    Invoke-WithConsent -Description "Install winget package '$Id'" -Action {
        _Install-WingetPackage -Id $Id -ExtraArgs $ExtraArgs
    }
}

function Install-WingetPackages {
    param(
        [string[]]$Ids
    )
    Invoke-WithConsent -Description "Install winget packages: $($Ids -join ' ')" -Action {
        foreach ($id in $Ids) {
            _Install-WingetPackage -Id $id
        }
    }
}
