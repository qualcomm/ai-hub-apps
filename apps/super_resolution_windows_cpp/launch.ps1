
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

param([switch]$NoDocker, [switch]$Clean,
      [Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArgs)
$ErrorActionPreference = "Stop"
$AppDir = $PSScriptRoot
$env:QAIHA_APP_ROOT = $AppDir
Set-Location $AppDir

# Drop the end-of-options separator the CLI forwards before passthrough args.
if ($AppArgs.Count -gt 0 -and $AppArgs[0] -eq "--") {
    $AppArgs = $AppArgs[1..($AppArgs.Count - 1)]
}

if (Test-Path "$AppDir\install_runtime.ps1") {
    Write-Host "::step::Installing runtime"
    & .\install_runtime.ps1
}

Write-Host "::step::Running super_resolution_windows_cpp natively"
& .\run.ps1 @AppArgs
exit $LASTEXITCODE
