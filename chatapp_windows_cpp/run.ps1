# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Exe = "$ScriptDir\ARM64\Release\ChatApp.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "ChatApp.exe not found at $Exe. Build the app first."
    exit 1
}

$GenieConfig = "$ScriptDir\genie_bundle\genie_config.json"
$BaseDir = "$ScriptDir\genie_bundle"
if (-not (Test-Path $GenieConfig)) {
    Write-Error "Genie config not found at $GenieConfig. Fetch the model first."
    exit 1
}

# The chat loop reads prompts from stdin, so nothing is piped in here -- typing
# at the prompt is the live input. Type 'exit' to quit.
& $Exe --genie-config $GenieConfig --base-dir $BaseDir @args
exit $LASTEXITCODE
