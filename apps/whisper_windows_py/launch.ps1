
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

# install_runtime.ps1 is skipped once it has succeeded for this exact app content;
# the marker holds a hash of everything the install depends on.
function Get-RuntimeStamp {
    $Files = @("install_runtime.ps1", "requirements*.txt", "versions.override.env", "scripts\versions.env")
    $Listing = (Get-Item -Path $Files -ErrorAction SilentlyContinue | Sort-Object FullName | ForEach-Object {
        "$((Get-FileHash -Algorithm SHA1 $_.FullName).Hash) $($_.Name)"
    }) -join "`n"
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Listing)
    (Get-FileHash -Algorithm SHA1 -InputStream ([System.IO.MemoryStream]::new($Bytes))).Hash
}

if (Test-Path "$AppDir\install_runtime.ps1") {
    $RuntimeMarker = "$AppDir\.qaiha_runtime_installed"
    if ($Clean -and (Test-Path $RuntimeMarker)) {
        Write-Host "-Clean: removing runtime marker at $RuntimeMarker"
        Remove-Item -Force $RuntimeMarker
    }
    $Stamp = Get-RuntimeStamp
    $Installed = if (Test-Path $RuntimeMarker) { (Get-Content -Raw $RuntimeMarker).Trim() } else { "" }
    if ($Installed -eq $Stamp) {
        Write-Host "::skip::Runtime already installed (marker at $RuntimeMarker matches $Stamp)"
    } else {
        if ($Installed -eq "") {
            Write-Host "No runtime marker at $RuntimeMarker; installing runtime"
        } else {
            Write-Host "Runtime marker at $RuntimeMarker is stale (recorded $Installed, current $Stamp); reinstalling runtime"
        }
        Write-Host "::step::Installing runtime"
        & .\install_runtime.ps1
        # $LASTEXITCODE is $null when the script ran no native command, which is not a
        # failure -- only a set, non-zero code is.
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            Write-Error "::error::install_runtime.ps1 failed (exit $LASTEXITCODE)"
            exit $LASTEXITCODE
        }
        Set-Content -Path $RuntimeMarker -Value $Stamp
        Write-Host "Wrote runtime marker $Stamp to $RuntimeMarker"
        Write-Host "::done::Installing runtime"
    }
}

$Script = if ($Test) { "test.ps1" } else { "run.ps1" }

Write-Host "::step::Running whisper_windows_py natively"
& ".\$Script" @AppArgs
exit $LASTEXITCODE
