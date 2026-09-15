# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$VenvPython = "$ScriptDir\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Run install_runtime.ps1 first."
    exit 1
}

$BaseUrl = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/stable_diffusion_windows_py/test"

$Tests = @(
    @{ Prompt = "A cat sitting on a bench";               OutputDir = "$env:TEMP\sd_cat";  RefUrl = "$BaseUrl/cat.png" },
    @{ Prompt = "A girl taking a walk at sunset";          OutputDir = "$env:TEMP\sd_girl"; RefUrl = "$BaseUrl/girl.png" },
    @{ Prompt = "A banyan tree with kids playing around";  OutputDir = "$env:TEMP\sd_tree"; RefUrl = "$BaseUrl/tree.png" }
)

$failed = 0
foreach ($test in $Tests) {
    New-Item -ItemType Directory -Force -Path $test.OutputDir | Out-Null

    Write-Host "Generating: $($test.Prompt)"
    & $VenvPython "$ScriptDir\demo.py" `
        --prompt $test.Prompt `
        --num-steps 20 `
        --seed 47 `
        --output-dir $test.OutputDir

    $generated = Join-Path $test.OutputDir "output.png"
    $reference = Join-Path $test.OutputDir "reference.png"

    Write-Host "Downloading reference image..."
    Invoke-WebRequest -Uri $test.RefUrl -OutFile $reference

    Write-Host "Comparing images..."
    & $VenvPython "$ScriptDir\compare_images.py" $generated $reference
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL: Image mismatch for prompt: '$($test.Prompt)'"
        $failed++
    }
}

if ($failed -gt 0) {
    Write-Error "$failed test(s) failed."
    exit 1
}

Write-Host "All image tests passed."
