# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# Installs the qai-hub-apps CLI from one of several sources:
#   source  -> editable install from the checked-out repo (cli/); no version needed
#   s3      -> published wheel from the self-hosted dev wheel index
#   staging -> published wheel from test.pypi.org
#   prod    -> published wheel from pypi.org
#
# With -Venv, installs into that venv's python; otherwise
# installs into the active environment.
#
# Usage:
#   . install_cli.ps1 -Source source [-Venv <path>]
#   . install_cli.ps1 -Source {s3|staging|prod} -Version <version> [-Venv <path>]
#
# Keep in sync with install_cli.sh.

param(
    [Parameter(Mandatory = $true)][ValidateSet("source", "s3", "staging", "prod")][string]$Source,
    [string]$Version = "",
    [string]$Venv = ""
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

# Tool-specific extra args (e.g. trusting a plain-HTTP index); set per source below.
$PipExtraArgs = @()
$UvExtraArgs = @()

# Install into a specific venv's python when -Venv is given, else the active env.
function Invoke-PipInstall {
    $uvAvailable = Get-Command uv -ErrorAction SilentlyContinue
    if ($Venv -ne "") {
        if ($uvAvailable) { uv pip install --python "$Venv\Scripts\python.exe" @UvExtraArgs @args }
        else { & "$Venv\Scripts\pip.exe" install @PipExtraArgs @args }
    } else {
        if ($uvAvailable) { uv pip install @UvExtraArgs @args }
        else { pip install @PipExtraArgs @args }
    }
}

if ($Source -eq "source") {
    Write-Host "Installing qai-hub-apps (editable) from cli/"
    Invoke-PipInstall -e "$(Get-RepoRoot)\cli\"
    return
}

switch ($Source) {
    "s3" {
        $indexHost = "qaihub-public-python-wheels.s3-website-us-west-2.amazonaws.com"
        $IndexUrl = "http://$indexHost/"
        # pip/uv ignore a plain-HTTP index unless the host is explicitly trusted.
        $PipExtraArgs = @("--trusted-host", $indexHost)
        $UvExtraArgs = @("--allow-insecure-host", $indexHost)
    }
    "staging" { $IndexUrl = "https://test.pypi.org/simple/" }
    "prod"    { $IndexUrl = "https://pypi.org/simple/" }
}

if ($Version -eq "") { throw "-Version is required for source '$Source'" }

$v = $Version -replace '^v', ''
Write-Host "Installing qai-hub-apps==$v from $Source ($IndexUrl)"
for ($i = 1; $i -le 10; $i++) {
    Invoke-PipInstall --pre --index-url $IndexUrl --extra-index-url "https://pypi.org/simple/" "qai-hub-apps==$v"
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "Attempt $i failed, retrying in 60s..."
    Start-Sleep -Seconds 60
    if ($i -eq 10) {
        Write-Error "Failed to install qai-hub-apps==$v after 10 attempts"
        exit 1
    }
}
