
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

param([switch]$NoDocker, [switch]$Clean, [switch]$Test,
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
    # $LASTEXITCODE is $null when the script ran no native command, which is not a
    # failure -- only a set, non-zero code is.
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        Write-Error "::error::install_runtime.ps1 failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    Write-Host "::done::Installing runtime"
}

$Script = if ($Test) { "test.ps1" } else { "run.ps1" }

Write-Host "::step::Running chatapp_windows_cpp natively"
& ".\$Script" @AppArgs
exit $LASTEXITCODE
