# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Build a docker base image and point the app build at it:
#
#   .\tools\build_base_images.ps1 android
#   $env:QAIHA_BASE_IMAGE = "qai-hub-apps-android-base:local"
#   qai-hub-apps run <app_id>
#
# android needs the daemon in linux-container mode, windows in windows-container
# mode.
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("android", "windows")]
    [string]$Platform,
    [string]$Tag = "local",
    [switch]$NoCache,
    [switch]$Push,
    [string]$Registry = "ghcr.io/qcom-ai-hub"
)
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\ci\common.ps1"

$RepoRoot = Get-RepoRoot
$Image = "qai-hub-apps-$Platform-base"
$BuildArgs = @()

if ($NoCache) { $BuildArgs += "--no-cache" }
if ($Push) { $Image = "$Registry/$Image" }
$Image = "${Image}:$Tag"

# The dockerfiles COPY from scripts/, so the context must be a directory with the
# shared scripts at that path -- not the repo root.
$Ctx = Join-Path ([System.IO.Path]::GetTempPath()) "qaiha-base-$([guid]::NewGuid().ToString('N').Substring(0, 8))"
try {
    New-Item -ItemType Directory -Path "$Ctx\scripts" -Force | Out-Null
    Copy-Item "$RepoRoot\apps\_shared\scripts\*" "$Ctx\scripts\" -Recurse -Force
    Copy-Item "$RepoRoot\tools\docker\$Platform.dockerfile" "$Ctx\Dockerfile" -Force

    Write-Host "::step::Building $Image"
    docker build @BuildArgs `
        --label "org.opencontainers.image.revision=$(git -C $RepoRoot rev-parse HEAD)" `
        -t $Image $Ctx
    if ($LASTEXITCODE -ne 0) {
        Write-Host "::error::Failed to build $Image."
        exit 1
    }
    Write-Host "::done::$Image"

    if ($Push) {
        Write-Host "::step::Pushing $Image"
        docker push $Image
        if ($LASTEXITCODE -ne 0) {
            Write-Host "::error::Failed to push $Image."
            exit 1
        }
        Write-Host "::done::$Image"
    }
}
finally {
    if (Test-Path $Ctx) { Remove-Item -Recurse -Force $Ctx }
}

Write-Host ""
if ($Push) {
    Write-Host "Pushed $Image"
} else {
    Write-Host "Export this before building an app:"
    Write-Host "  `$env:QAIHA_BASE_IMAGE = `"$Image`""
}
