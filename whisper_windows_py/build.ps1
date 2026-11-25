
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

param([switch]$NoDocker, [switch]$Clean)
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "::skip::Nothing to build for whisper_windows_py; run it directly (see the app README)."
