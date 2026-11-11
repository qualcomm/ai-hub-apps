# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# install_runtime.ps1 is skipped once it has succeeded for this exact app content;
# the marker holds a hash of everything the install depends on.
#
# Functions:
#   Get-RuntimeStamp -Files <string[]>
#       Return a hash of the given files that exist.
#   Install-Runtime -Marker <path> -Stamp <str>
#       Run install_runtime.ps1 unless <Marker> already holds <Stamp>.
#
# Usage: . runtime_utils.ps1
# ---------------------------------------------------------------------

function Get-RuntimeStamp {
    param([string[]]$Files)
    $Listing = (Get-Item -Path $Files -ErrorAction SilentlyContinue | Sort-Object FullName | ForEach-Object {
        "$((Get-FileHash -Algorithm SHA1 $_.FullName).Hash) $($_.Name)"
    }) -join "`n"
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Listing)
    (Get-FileHash -Algorithm SHA1 -InputStream ([System.IO.MemoryStream]::new($Bytes))).Hash
}

function Install-Runtime {
    param([string]$Marker, [string]$Stamp)
    $Installed = if (Test-Path $Marker) { (Get-Content -Raw $Marker).Trim() } else { "" }
    if ($Installed -eq $Stamp) {
        Write-Host "::skip::Runtime already installed (marker at $Marker matches $Stamp)"
        return
    }
    if ($Installed -eq "") {
        Write-Host "No runtime marker at $Marker; installing runtime"
    } else {
        Write-Host "Runtime marker at $Marker is stale (recorded $Installed, current $Stamp); reinstalling runtime"
    }
    Write-Host "::step::Installing runtime"
    & .\install_runtime.ps1
    # $LASTEXITCODE is $null when the script ran no native command, which is not a
    # failure -- only a set, non-zero code is.
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        Write-Error "::error::install_runtime.ps1 failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    Set-Content -Path $Marker -Value $Stamp
    Write-Host "Wrote runtime marker $Stamp to $Marker"
    Write-Host "::done::Installing runtime"
}
