# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Create a Python virtual environment and install qai_hub_apps_test.
#
# Usage:
#   . tools/setup_env.ps1 [-Venv <path>] [-Python <exe>] [-Extras <extra>] [-WithCli]
#
# Defaults:
#   -Venv    qaiha-dev
#   -Python  python
#   -Extras  dev
#
# Available extras:
#   dev        Full test install: pytest, qai-hub-models-cli, boto3, etc. (default)
#   precommit  Light install: pre-commit + mypy only (for CI lint checks)
#
# Flags:
#   -WithCli      Also install the qai-hub-apps CLI package (cli/)
#   -WithQdcSdk   Also download and install the Qualcomm Device Cloud SDK wheel

param(
    [string]$Venv = "qaiha-dev",
    [string]$Python = "python",
    [string]$Extras = "dev",
    [switch]$WithCli,
    [switch]$WithQdcSdk
)

$ErrorActionPreference = "Stop"

$RepoRoot = git rev-parse --show-toplevel

if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtual environment at $Venv using $Python"
    & $Python -m venv $Venv
} else {
    Write-Host "Virtual environment already exists at $Venv"
}

$InstallTarget = "$RepoRoot\tools\python\[$Extras]"

$uvAvailable = Get-Command uv -ErrorAction SilentlyContinue
if ($uvAvailable) {
    uv pip install --python "$Venv\Scripts\python.exe" -e $InstallTarget
} else {
    & "$Venv\Scripts\pip.exe" install -e $InstallTarget
}

if ($WithCli) {
    Write-Host "Installing CLI package (cli/)..."
    $uvAvailable = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvAvailable) {
        uv pip install --python "$Venv\Scripts\python.exe" -e "$RepoRoot\cli\"
    } else {
        & "$Venv\Scripts\pip.exe" install -e "$RepoRoot\cli\"
    }
}

if ($WithQdcSdk) {
    Write-Host "Downloading and installing QDC SDK wheel..."
    $QdcTmpDir = Join-Path $env:TEMP "qdc_wheel_$([System.IO.Path]::GetRandomFileName())"
    & "$RepoRoot\tools\ci\download-qdc-wheel.ps1" -DestDir $QdcTmpDir
    $wheel = (Get-ChildItem -Path $QdcTmpDir -Filter *.whl)[0].FullName
    $uvAvailable = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvAvailable) {
        uv pip install --python "$Venv\Scripts\python.exe" $wheel
    } else {
        & "$Venv\Scripts\pip.exe" install $wheel
    }
    Remove-Item -Recurse -Force $QdcTmpDir
}

Write-Host ""
Write-Host "Done. Activate with: . $Venv\Scripts\Activate.ps1"
