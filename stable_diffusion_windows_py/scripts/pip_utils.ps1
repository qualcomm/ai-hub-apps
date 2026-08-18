# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Windows pip/venv installation utilities.
#
# Functions:
#   Install-PipDeps [-VenvDir <path>] [-Python <exe>] [-Packages <string[]>] [-ExtraArgs <string[]>]
#       Create a .venv (if needed) and install packages or requirements files via uv.
#       -VenvDir <path>  : venv directory (default: $PWD\.venv)
#       -Python <exe>    : Python executable to use for venv creation (default: py -<version>)
#       -Packages <str[]>: package specs or -r requirements.txt entries
#       -ExtraArgs <str[]>: extra flags passed directly to uv pip install
#
# Usage: . pip_utils.ps1
# ---------------------------------------------------------------------
$_PipUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_PipUtilsDir\load_versions.ps1"
. "$_PipUtilsDir\interactive.ps1"
. "$_PipUtilsDir\winget_utils.ps1"

function _Install-PipDeps {
    param(
        [string]$VenvDir = "",
        [string]$Python = "",
        [string[]]$Packages = @(),
        [string[]]$ExtraArgs = @()
    )
    if ($VenvDir -eq "") {
        $VenvDir = Join-Path $PWD ".venv"
    }
    $ver = $PYTHON_VERSION
    $majorMinor = ($ver -split "\.")[ 0..1] -join "."
    if ($Python -eq "") {
        $py = Resolve-InstalledExe -Name "py.exe"
        if (-not $py) {
            Write-Error "py launcher not found. Run Install-Python first, or pass -Python <exe>."
            exit 1
        }
        $PythonExe = $py
        $PythonArgs = @("-$majorMinor")
    } else {
        $PythonExe = $Python
        $PythonArgs = @()
    }
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"

    if (-not (Test-Path $VenvPython)) {
        Write-Host "::step::Creating virtual environment at $VenvDir"
        & $PythonExe @PythonArgs -m venv $VenvDir
        Write-Host "::done::virtual environment"
    }

    Write-Host "::step::Installing uv"
    & $VenvPython -m pip install --quiet uv

    $uvExe = Join-Path (Split-Path $VenvPython) "uv.exe"
    Write-Host "::step::Installing Python dependencies"
    & $uvExe pip install --python $VenvPython @Packages @ExtraArgs
    Write-Host "::done::pip install"
}

function Install-PipDeps {
    param(
        [string]$VenvDir = "",
        [string]$Python = "",
        [string[]]$Packages = @(),
        [string[]]$ExtraArgs = @()
    )
    Invoke-WithConsent -Description "Create/populate a virtual environment and install Python dependencies" -Action {
        _Install-PipDeps -VenvDir $VenvDir -Python $Python -Packages $Packages -ExtraArgs $ExtraArgs
    }
}
