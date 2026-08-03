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
# Override the version before calling:
#   $PYTHON_VERSION = "3.12"
#   Install-Python
#
# Usage: . python_utils.ps1
# ---------------------------------------------------------------------
$_PythonUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_PythonUtilsDir\load_versions.ps1"
. "$_PythonUtilsDir\winget_utils.ps1"
. "$_PythonUtilsDir\interactive.ps1"

function _Install-Python {
    $ver = $PYTHON_VERSION
    $majorMinor = ($ver -split "\.")[ 0..1] -join "."
    Install-WingetPackage -Id "Python.Python.$majorMinor" -ExtraArgs @("--source", "winget")

    Write-Host "::step::Bootstrapping uv"
    py -$majorMinor -m pip install --quiet uv
    Write-Host "::done::uv"
}

function Install-Python {
    Invoke-WithConsent -Description "Install Python $PYTHON_VERSION via winget" -Action {
        _Install-Python
    }
}
