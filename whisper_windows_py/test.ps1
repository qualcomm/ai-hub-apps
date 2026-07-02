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

$AudioUrl = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/hf_whisper_asr_shared/v1/audio/fox.wav"
$AudioFile = "$env:TEMP\fox.wav"
# Expected transcription of the sample audio (case-insensitive substring match).
$Expected = "quick brown fox jumps over the lazy"

Write-Host "Downloading sample audio..."
Invoke-WebRequest -Uri $AudioUrl -OutFile $AudioFile

Write-Host "Transcribing..."
$output = & $VenvPython "$ScriptDir\demo.py" --audio-file $AudioFile | Out-String
Write-Host $output

if ($output -match [regex]::Escape($Expected)) {
    Write-Host "PASS: Transcription contains expected text '$Expected'."
} else {
    Write-Error "FAIL: Transcription did not contain expected text '$Expected'."
    exit 1
}
