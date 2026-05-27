# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Windows QAIRT SDK installation utilities.
#
# Functions:
#   Install-Qairt [-Force]
#       Download and extract the QAIRT SDK to $QAIRT_PATH. Skips if already
#       installed unless -Force is passed. After dot-sourcing this script,
#       $QAIRT_PATH holds the SDK root directory.
#
# Usage: . qairt_utils.ps1
# ---------------------------------------------------------------------
$_QairtUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_QairtUtilsDir\load_versions.ps1"

$QAIRT_ROOT = "C:\Qualcomm\AIStack\QAIRT"
$QAIRT_PATH = "$QAIRT_ROOT\$QAIRT_SDK_FULL_VERSION"
$env:QAIRT_ROOT = $QAIRT_ROOT
$env:QAIRT_PATH = $QAIRT_PATH

function Install-Qairt {
    param([switch]$Force)

    if ($env:QAIRT_INSTALL_SKIP -eq "true") {
        Write-Host "::skip::QAIRT SDK install deferred (QAIRT_INSTALL_SKIP=true)"
        return
    }

    if ((Test-Path $QAIRT_PATH) -and (Get-ChildItem $QAIRT_PATH -Force | Select-Object -First 1) -and -not $Force) {
        Write-Host "::skip::QAIRT SDK already installed at $QAIRT_PATH"
        return
    }

    $url = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QAIRT_SDK_FULL_VERSION}/v${QAIRT_SDK_FULL_VERSION}.zip"
    $tmpZip = Join-Path $env:TEMP "qairt_$([System.IO.Path]::GetRandomFileName()).zip"

    Write-Host "::step::Downloading QAIRT SDK $QAIRT_SDK_FULL_VERSION"
    Write-Host "   URL: $url"
    curl.exe -L -o $tmpZip $url
    Write-Host "::done::download"

    Write-Host "::step::Extracting QAIRT SDK"
    $tmpDir = Join-Path $env:TEMP "qairt_extract_$([System.IO.Path]::GetRandomFileName())"
    New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
    tar -xf $tmpZip -C $tmpDir
    Remove-Item $tmpZip

    $extracted = Join-Path $tmpDir "qairt\$QAIRT_SDK_FULL_VERSION"
    New-Item -ItemType Directory -Path $QAIRT_ROOT -Force | Out-Null
    if (Test-Path $QAIRT_PATH) { Remove-Item -Recurse -Force $QAIRT_PATH }
    Move-Item $extracted $QAIRT_PATH
    Remove-Item -Recurse -Force $tmpDir
    Write-Host "::done::QAIRT SDK installed at $QAIRT_PATH"
}
