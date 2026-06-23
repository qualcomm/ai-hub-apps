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

$QdcSdkUrl = "https://softwarecenter.qualcomm.com/api/download/software/tools/Qualcomm_Device_Cloud_SDK/Windows/0.2.3/qualcomm_device_cloud_sdk-0.2.3.zip"
$QdcSdkSha256 = "ff14974c134dae8064ba15a8d78ebc62c480573c947c612f929718bd1c406d27"

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
