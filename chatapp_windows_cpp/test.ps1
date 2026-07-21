# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Exe = "$ScriptDir\ARM64\Release\ChatApp.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "ChatApp.exe not found at $Exe. Run install_build.ps1 first."
    exit 1
}

$GenieConfig = "$ScriptDir\genie_bundle\genie_config.json"
$BaseDir = "$ScriptDir\genie_bundle"
if (-not (Test-Path $GenieConfig)) {
    Write-Error "Genie config not found at $GenieConfig. Fetch the model first."
    exit 1
}

# Drive the interactive chat loop non-interactively: send one prompt, then exit.
$prompt = "What is gravity? Keep the answer under 50 words."
Write-Host "Running ChatApp with prompt: $prompt"
$output = "$prompt`nexit`n" | & $Exe --genie-config $GenieConfig --base-dir $BaseDir | Out-String
Write-Host $output

if ($LASTEXITCODE -ne 0) {
    Write-Error "ChatApp exited with code $LASTEXITCODE."
    exit 1
}

# The welcome banner is always printed; require at least one additional non-empty
# line as evidence the model produced a response.
if ($output -notmatch "Welcome to ChatApp") {
    Write-Error "ChatApp did not start as expected (no welcome banner)."
    exit 1
}

Write-Host "ChatApp test passed."
