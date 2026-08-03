# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Build-time setup: installs Visual Studio Build Tools, vcpkg (for OpenCV) and
# NuGet, restores the NuGet packages, and exports $env:MSBUILD_EXE for the
# caller to invoke MSBuild. OpenCV (vcpkg manifest) is restored during the build.
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\scripts\winget_utils.ps1"
. "$PSScriptRoot\scripts\msvc_utils.ps1"
. "$PSScriptRoot\scripts\vcpkg_utils.ps1"

# Install Visual Studio Build Tools (C++ ARM64); sets $env:MSBUILD_EXE.
Install-MSVC
Install-WingetPackage -Id "Microsoft.Git" -ExtraArgs @("--source", "winget")

# Install vcpkg (OpenCV via manifest) and NuGet, then restore NuGet packages.
Install-Vcpkg
Install-NuGet
& $env:NUGET_EXE restore ObjectDetection.sln -MSBuildPath (Split-Path $env:MSBUILD_EXE)
