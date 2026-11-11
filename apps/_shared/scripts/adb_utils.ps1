# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# adb helpers for the generated Android launch.ps1.
#
# Functions:
#   Select-AdbDevice
#       Return the connected device's serial, prompting when there are several.
#       Exits if none is connected or the selection is invalid.
#
# Usage: . adb_utils.ps1
# ---------------------------------------------------------------------

function Select-AdbDevice {
    $Devices = @(adb devices | Select-Object -Skip 1 | ForEach-Object {
        $Fields = $_ -split '\s+'
        if ($Fields.Count -ge 2 -and $Fields[1] -eq "device") { $Fields[0] }
    })

    if ($Devices.Count -eq 0) {
        Write-Host "::error::No Android device connected. Connect a device with USB debugging enabled."
        exit 1
    } elseif ($Devices.Count -eq 1) {
        return $Devices[0]
    }
    Write-Host "Connected devices:"
    for ($i = 0; $i -lt $Devices.Count; $i++) {
        Write-Host ("  {0}) {1}" -f ($i + 1), $Devices[$i])
    }
    $Choice = Read-Host "Select a device [1-$($Devices.Count)]"
    if ($Choice -notmatch '^\d+$' -or [int]$Choice -lt 1 -or [int]$Choice -gt $Devices.Count) {
        Write-Host "::error::Invalid selection '$Choice'."
        exit 1
    }
    $Devices[[int]$Choice - 1]
}
