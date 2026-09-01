# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# Downloads the QDC SDK zip, verifies its checksum, extracts the wheel,
# and copies it to the destination directory.
#
# Usage: . download-qdc-wheel.ps1 -DestDir <destination_dir>
#
# Keep QDC_SDK_URL / QDC_SDK_SHA256 in sync with download-qdc-wheel.sh.

param(
    [Parameter(Mandatory = $true)][string]$DestDir
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

$QdcSdkUrl = "https://softwarecenter.qualcomm.com/api/download/software/tools/Qualcomm_Device_Cloud_SDK/All/0.4.1/qualcomm_device_cloud_sdk-0.4.1.zip"
$QdcSdkSha256 = "716a862ce64f9146078cd0b7b7ab18d2672520e068345accbb094e848cc22cfb"

$tmpZip = Join-Path $env:TEMP "qualcomm_device_cloud_sdk.zip"
$tmpDir = Join-Path $env:TEMP "qualcomm_device_cloud_sdk"

Invoke-DownloadAndVerify -Url $QdcSdkUrl -Dest $tmpZip -Sha256 $QdcSdkSha256

if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
Expand-Archive -Path $tmpZip -DestinationPath $tmpDir -Force

$wheels = @(Get-ChildItem -Path $tmpDir -Filter *.whl)
if ($wheels.Count -ne 1) {
    Write-Error "Expected exactly one .whl, found $($wheels.Count)."
    exit 1
}

New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
Copy-Item $wheels[0].FullName -Destination $DestDir

Remove-Item -Recurse -Force $tmpDir
Remove-Item -Force $tmpZip
