
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

param([switch]$NoDocker, [switch]$Clean)
$ErrorActionPreference = "Stop"

function Assert-Success {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "::error::$What failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

$AppDir = $PSScriptRoot
Set-Location $AppDir

. ..\_shared\scripts\docker_utils.ps1

if ($NoDocker) {
    Write-Host "::error::Android apps require Docker to build (no native build)."
    exit 1
}

if (-not (Test-Path "$AppDir\Dockerfile")) {
    Write-Host "::error::No Dockerfile found for image_classification_android; it cannot be built."
    exit 1
}

# install_build.sh and the gradle step source /app/scripts/*, which only exists in
# a fetched/bundled app.
if (-not (Test-Path "$AppDir\scripts")) {
    Write-Host "::error::No scripts\ directory found for image_classification_android. Docker builds need a bundled app; use 'qai-hub-apps fetch image_classification_android'."
    exit 1
}

$ImageTag = Get-DockerImageTag -Prefix "aiha-build" -AppDir $AppDir
$ContainerName = "$ImageTag-container"

# The Android toolchain lives in a prebuilt base image, so this build is just a
# pull.
$BaseImage = Resolve-BaseImage -Pinned "ghcr.io/qcom-ai-hub/qai-hub-apps-android-base:sha-f3ce55a28d93"
$BuildArgs = @("--build-arg", "BASE_IMAGE=$BaseImage")

# -Clean tears down prior build state (image, container, host-side outputs).
# --no-cache only invalidates this app's trivial FROM layer, not the base image;
# to force a fresh base, 'docker rmi' it or bump the tag in versions.env.
if ($Clean) {
    Write-Host "::step::Cleaning prior build outputs, docker image and container"
    $BuildArgs += "--no-cache"
    if (Test-Path ".\build\outputs") { Remove-Item -Recurse -Force ".\build\outputs" }
    try { docker rm -f $ContainerName 2>$null | Out-Null } catch {}
    try { docker rmi $ImageTag 2>$null | Out-Null } catch {}
    Write-Host "::done::clean"
}

Write-Host "::step::Building Docker image from $BaseImage"
docker build @BuildArgs -t $ImageTag .
if ($LASTEXITCODE -ne 0) {
    Write-Host "::error::Failed to build the image for image_classification_android. If '$BaseImage' could not be pulled, check network access to it, or set QAIHA_BASE_IMAGE to a base image you built locally."
    exit 1
}
Write-Host "::done::Docker image"

# Reuse this app directory's container, or create it.
Start-Container -Name $ContainerName -Image $ImageTag -CreateArgs @("-v", "${AppDir}:/app")

try {
    Install-QcCaCerts -Container $ContainerName -JavaKeystore

    if (Test-Path "$AppDir\install_build.sh") {
        Write-Host "::step::Installing build dependencies in $ContainerName"
        docker exec -w /app $ContainerName bash install_build.sh
        Assert-Success "install_build.sh"
        Write-Host "::done::Installing build dependencies"
    }

    Write-Host "::step::Building APKs (gradle assembleDebug assembleAndroidTest)"
    # 'source' rather than '.' so the bundler's PowerShell dot-source pattern
    # does not try to rewrite these container-side paths.
    docker exec -w /app $ContainerName bash -c '
    set -euo pipefail
    source /app/scripts/android_utils.sh
    if [ -f /app/scripts/qairt_utils.sh ]; then
        source /app/scripts/qairt_utils.sh
    fi
    gradle assembleDebug assembleAndroidTest'
    Assert-Success "gradle"
    Write-Host "::done::APKs built into $AppDir\build\outputs"
}
finally {
    # Stop the container after the build so it holds no resources between builds.
    try { docker stop $ContainerName 2>$null | Out-Null } catch {}
}
