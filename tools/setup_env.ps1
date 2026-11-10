# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Create a Python virtual environment and install the qai-hub-apps CLI package,
# which includes the internal tooling (qai_hub_apps._internal).
#
# Usage:
#   . tools/setup_env.ps1 [-Venv <path>] [-Python <exe>] [-Recreate] [-WithCli] [-WithQdcSdk]
#
# Defaults:
#   -Venv    qaiha-dev
#   -Python  python
#
# -Python takes an interpreter executable name or path (e.g. python3.12.exe,
# C:\Python312\python.exe), not a bare version like 3.12.
#
# Flags:
#   -Recreate     Delete and recreate the venv if it already exists
#   -WithCli      Also install the qai-hub-apps CLI package (cli/)
#   -WithQdcSdk   Also download and install the Qualcomm Device Cloud SDK wheel
#
# If the venv already exists with a different Python major.minor than -Python,
# the script fails; pass -Recreate or a different -Venv.

param(
    [string]$Venv = "qaiha-dev",
    [string]$Python = "python",
    [switch]$Recreate,
    [switch]$WithCli,
    [switch]$WithQdcSdk
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\ci\common.ps1"
$RepoRoot = Get-RepoRoot

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python executable not found: $Python (-Python takes an executable, e.g. python3.12.exe)"
}

function Get-PyVersion([string]$Exe) {
    $version = & $Exe -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    return "$version".Trim()
}

$VenvPython = "$Venv\Scripts\python.exe"

if ((Test-Path $Venv) -and -not $Recreate) {
    $VenvVersion = $null
    if (Test-Path $VenvPython) { $VenvVersion = Get-PyVersion $VenvPython }
    if (-not $VenvVersion) {
        throw "$Venv exists but is not a working virtual environment. Re-run with -Recreate, or delete $Venv."
    }
    $RequestedVersion = Get-PyVersion $Python
    if ($VenvVersion -ne $RequestedVersion) {
        throw "$Venv uses Python $VenvVersion, but $Python is Python $RequestedVersion. Re-run with -Recreate, pass a different -Venv, or delete $Venv."
    }
    Write-Host "Virtual environment already exists at $Venv (Python $VenvVersion)"
} else {
    if (Test-Path $Venv) {
        Write-Host "Removing existing virtual environment at $Venv"
        Remove-Item -Recurse -Force $Venv
    }
    Write-Host "Creating virtual environment at $Venv using $Python"
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment at $Venv" }
}

$InstallTarget = "$RepoRoot\cli\[dev]"

$uvAvailable = Get-Command uv -ErrorAction SilentlyContinue
if ($uvAvailable) {
    uv pip install --python $VenvPython -e $InstallTarget
} else {
    & "$Venv\Scripts\pip.exe" install -e $InstallTarget
}

if ($WithCli) {
    Write-Host "Note: -WithCli is a no-op; the CLI package is installed above."
}

# The Java/Kotlin pre-commit hooks need a JVM toolchain
# (JDK + google-java-format + ktlint)
. "$RepoRoot\tools\lint\install_lint_tools.ps1"

if ($WithQdcSdk) {
    Write-Host "Downloading and installing QDC SDK wheel..."
    $QdcTmpDir = Join-Path $env:TEMP "qdc_wheel_$([System.IO.Path]::GetRandomFileName())"
    & "$RepoRoot\tools\ci\download-qdc-wheel.ps1" -DestDir $QdcTmpDir
    $wheel = (Get-ChildItem -Path $QdcTmpDir -Filter *.whl)[0].FullName
    $uvAvailable = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvAvailable) {
        uv pip install --python $VenvPython $wheel
    } else {
        & "$Venv\Scripts\pip.exe" install $wheel
    }
    Remove-Item -Recurse -Force $QdcTmpDir
}

Write-Host ""
Write-Host "Done. Activate with: . $Venv\Scripts\Activate.ps1"
