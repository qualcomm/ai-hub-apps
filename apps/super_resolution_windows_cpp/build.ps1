
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

$Sln = "SuperResolution.sln"

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

try {
    Write-Host "::step::Building Docker image"
    docker build --build-arg BUILD_TYPE=build -t $ImageTag .
    Assert-Success "docker build"
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
    docker run --name $ContainerName $ImageTag `
        powershell -Command ". ./install_build.ps1; & `$env:MSBUILD_EXE $Sln /p:Configuration=Release /p:Platform=ARM64; exit `$LASTEXITCODE"
    Assert-Success "docker run (MSBuild)"

    docker cp "${ContainerName}:C:\app\ARM64" .
    Assert-Success "docker cp"
    Write-Host "::done::ARM64 binaries built into $AppDir\ARM64"
}
finally {
    # The container is transient; remove it. The image is kept for cache reuse
    # (removed only by -Clean).
    try { docker rm -f $ContainerName 2>$null | Out-Null } catch {}
}
