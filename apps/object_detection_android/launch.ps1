
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

param([switch]$NoDocker, [switch]$Docker, [switch]$Clean, [switch]$Test,
      [Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArgs)
$ErrorActionPreference = "Stop"

$AppDir = $PSScriptRoot
Set-Location $AppDir

. ..\_shared\scripts\exit_codes.ps1
. ..\_shared\scripts\adb_utils.ps1

$Package = "com.quicinc.objectdetection"
$Apk = "build\outputs\apk\debug\app-debug.apk"
$TestApk = "build\outputs\apk\androidTest\debug\app-debug-androidTest.apk"
$Runner = "com.quicinc.objectdetection.test/androidx.test.runner.AndroidJUnitRunner"

if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
    Write-Host "::error::adb not found on PATH. Install the Android platform-tools."
    exit 1
}

if (-not (Test-Path $Apk)) {
    Write-Host "::error::APK not found at $Apk for object_detection_android. Build it first."
    exit $QaihaExitBuildRequired
}
if ($Test -and -not (Test-Path $TestApk)) {
    Write-Host "::error::Test APK not found at $TestApk for object_detection_android. Build it first."
    exit $QaihaExitBuildRequired
}

$Serial = Select-AdbDevice

Write-Host "::step::Installing $Apk on $Serial"
adb -s $Serial install -r -t $Apk
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Test) {
    Write-Host "::step::Installing $TestApk on $Serial"
    adb -s $Serial install -r -t $TestApk
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "::step::Running instrumentation tests for object_detection_android"
    $Output = (adb -s $Serial shell am instrument -w -r $Runner 2>&1) -join "`n"
    $Rc = $LASTEXITCODE
    Write-Host $Output
    if ($Rc -ne 0 -or $Output -match "INSTRUMENTATION_FAILED|FAILURES") {
        Write-Host "::error::Instrumentation tests failed for object_detection_android."
        exit 1
    }
    Write-Host "::done::test"
} else {
    Write-Host "::step::Launching object_detection_android ($Package)"
    adb -s $Serial shell monkey -p $Package -c android.intent.category.LAUNCHER 1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "::done::run"
}
