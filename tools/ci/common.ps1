# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Shared CI PowerShell utilities.
#
# Functions:
#   Get-RepoRoot
#       Return the absolute path to the repository root.
#   Invoke-DownloadAndVerify -Url <url> -Dest <dest_file> [-Sha256 <sha256>]
#       Download <url> to <dest_file>. If <sha256> is provided, verifies the
#       checksum and throws if it does not match.
#
# Usage: . common.ps1
# ---------------------------------------------------------------------

function Get-RepoRoot {
    git rev-parse --show-toplevel
}

function Invoke-DownloadAndVerify {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Dest,
        [string]$Sha256 = ""
    )

    Write-Host "Downloading $(Split-Path -Leaf $Dest)"
    Write-Host "   URL: $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Dest

    if ($Sha256 -ne "") {
        $actual = (Get-FileHash -Algorithm SHA256 -Path $Dest).Hash.ToLower()
        if ($actual -ne $Sha256.ToLower()) {
            throw "Checksum mismatch for $Dest. Expected $Sha256, got $actual."
        }
    }
    Write-Host "Downloaded and verified"
}
