# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Windows Python installation utilities.
#
# Functions:
#   Install-Python
#       Install the Python version given by $PYTHON_VERSION (default from
#       versions.env) via winget. Also bootstraps uv.
#
#
# Usage: . python_utils.ps1
# ---------------------------------------------------------------------
$_PythonUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_PythonUtilsDir\load_versions.ps1"
. "$_PythonUtilsDir\winget_utils.ps1"
. "$_PythonUtilsDir\interactive.ps1"

# uv is the last thing _Install-Python does, so its presence for this interpreter
# means the whole install completed.
function Test-PythonInstalled {
    $py = Resolve-InstalledExe -Name "py.exe"
    if (-not $py) { return $false }
    $majorMinor = ($PYTHON_VERSION -split "\.")[ 0..1] -join "."
    try { & $py -$majorMinor -m uv --version 2>$null | Out-Null } catch { return $false }
    return $LASTEXITCODE -eq 0
}

function _Install-Python {
    $ver = $PYTHON_VERSION
    $majorMinor = ($ver -split "\.")[ 0..1] -join "."
    Install-WingetPackage -Id "Python.Python.$majorMinor" -ExtraArgs @("--source", "winget")

    # winget just added the `py` launcher to the persistent PATH, but this process
    # can't see it yet, so resolve it by absolute path from the registry PATH.
    $py = Resolve-InstalledExe -Name "py.exe"
    if (-not $py) {
        Write-Error "py launcher not found after Python install."
        exit 1
    }

    Write-Host "::step::Bootstrapping uv"
    & $py -$majorMinor -m pip install --quiet uv
    Write-Host "::done::uv"
}

function Install-Python {
    if (Test-PythonInstalled) {
        Write-Host "::skip::Python $PYTHON_VERSION already installed"
        return
    }
    Invoke-WithConsent -Description "Install Python $PYTHON_VERSION via winget" -Action {
        _Install-Python
    }
}
