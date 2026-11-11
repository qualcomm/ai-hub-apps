
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

. ..\_shared\scripts\runtime_utils.ps1

# Drop the end-of-options separator the CLI forwards before passthrough args.
if ($AppArgs.Count -gt 0 -and $AppArgs[0] -eq "--") {
    $AppArgs = $AppArgs[1..($AppArgs.Count - 1)]
}

if (Test-Path "$AppDir\install_runtime.ps1") {
    $RuntimeMarker = "$AppDir\.qaiha_runtime_installed"
    if ($Clean -and (Test-Path $RuntimeMarker)) {
        Write-Host "-Clean: removing runtime marker at $RuntimeMarker"
        Remove-Item -Force $RuntimeMarker
    }
    Install-Runtime -Marker $RuntimeMarker -Stamp (Get-RuntimeStamp -Files @("install_runtime.ps1", "requirements*.txt", "versions.override.env", "scripts\versions.env"))
}

$Script = if ($Test) { "test.ps1" } else { "run.ps1" }

Write-Host "::step::Running whisper_windows_py natively"
& ".\$Script" @AppArgs
exit $LASTEXITCODE
