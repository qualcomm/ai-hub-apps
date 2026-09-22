
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

# Builds the Windows C++ app's ARM64 binaries. Docker (default) uses a Windows
# container image and requires a Windows Docker host; -NoDocker builds natively
# with the host MSBuild. -Clean removes prior build artifacts first.
param([switch]$NoDocker, [switch]$Clean)
$ErrorActionPreference = "Stop"

. ..\_shared\scripts\interactive.ps1

function Assert-Success {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "::error::$What failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

$AppDir = $PSScriptRoot
Set-Location $AppDir

$Sln = "ObjectDetection.sln"

if ($NoDocker) {
    . ./install_build.ps1
    Assert-Success "install_build.ps1"
    if ($Clean) {
        Write-Host "::step::Cleaning prior build outputs"
        & $env:MSBUILD_EXE $Sln /t:Clean /p:Configuration=Release /p:Platform=ARM64
        Assert-Success "MSBuild clean"
        Write-Host "::done::clean"
    }
    Write-Host "::step::Building ARM64 binaries (MSBuild)"
    & $env:MSBUILD_EXE $Sln /p:Configuration=Release /p:Platform=ARM64
    Assert-Success "MSBuild"
    Write-Host "::done::ARM64 binaries built into $AppDir\ARM64"
    exit 0
}

# Derive unique image/container names from the app directory so two copies of
# the same app in different directories never collide.
$Sha1 = [System.Security.Cryptography.SHA1]::Create()
$Bytes = [System.Text.Encoding]::UTF8.GetBytes($AppDir)
$Hash = ([System.BitConverter]::ToString($Sha1.ComputeHash($Bytes)) -replace '-', '').ToLower().Substring(0, 12)
$ImageTag = "aiha-build-$(Split-Path $AppDir -Leaf)-$Hash"
$ContainerName = "$ImageTag-container"

# -Clean tears down prior build state (image, container, host-side outputs) and
# rebuilds the image from scratch. Without it, the image is left in place so the
# next build reuses its cache.
if ($Clean) {
    Write-Host "::step::Cleaning prior build outputs, docker image and container"
    if (Test-Path ".\ARM64") { Remove-Item -Recurse -Force ".\ARM64" }
    try { docker rm -f $ContainerName 2>$null | Out-Null } catch {}
    try { docker rmi $ImageTag 2>$null | Out-Null } catch {}
    Write-Host "::done::clean"
}

if (-not (Test-Path "$AppDir\Dockerfile")) {
    Write-Error "::error::No Dockerfile found for object_detection_windows_cpp. Re-run with -NoDocker to build natively."
    exit 1
}

# install_build.ps1 sources C:\app\scripts\*, which only exists in a fetched/bundled
# app. The image build itself needs nothing from the context.
if (-not (Test-Path "$AppDir\scripts")) {
    Write-Error "::error::No scripts\ directory found for object_detection_windows_cpp. Docker builds need a bundled app; use 'qai-hub-apps fetch object_detection_windows_cpp'."
    exit 1
}

try {
    # The MSVC toolchain lives in a prebuilt base image, so this build is just a
    # pull. QAIHA_BASE_IMAGE overrides it, e.g. to point at a locally built base.
    if ($env:QAIHA_BASE_IMAGE) {
        $BaseImage = $env:QAIHA_BASE_IMAGE
    } else {
        $BaseImage = "ghcr.io/qcom-ai-hub/qai-hub-apps-windows-base:sha-72a927d1095e"
    }

    Write-Host "::step::Building Docker image from $BaseImage"
    docker build --build-arg "BASE_IMAGE=$BaseImage" -t $ImageTag .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "::error::Failed to build the image for object_detection_windows_cpp. If '$BaseImage' could not be pulled, check network access to it, or set QAIHA_BASE_IMAGE to a base image you built locally from tools/docker/windows.dockerfile. Alternatively re-run with -NoDocker to build natively."
        exit 1
    }
    Write-Host "::done::Docker image"

    Write-Host "::step::Building ARM64 binaries (MSBuild in container)"
    # A container from a prior run may still hold this name (e.g. after a hard
    # kill that skipped the cleanup below). Ask before removing it.
    $exists = $false
    try { docker container inspect $ContainerName 2>$null | Out-Null; $exists = $true } catch {}
    if ($exists) {
        Invoke-WithConsent -Description "A container named '$ContainerName' already exists (likely a leftover from a previous run). Remove it?" -Action {
            docker rm -f $ContainerName 2>$null | Out-Null
        }
    }
    docker run --name $ContainerName -v "${AppDir}:C:\app" $ImageTag `
        powershell -Command ". ./install_build.ps1; & `$env:MSBUILD_EXE $Sln /p:Configuration=Release /p:Platform=ARM64; exit `$LASTEXITCODE"
    Assert-Success "docker run (MSBuild)"

    Write-Host "::done::ARM64 binaries built into $AppDir\ARM64"
}
finally {
    # The container is transient; remove it. The image is kept for cache reuse
    # (removed only by -Clean).
    try { docker rm -f $ContainerName 2>$null | Out-Null } catch {}
}
