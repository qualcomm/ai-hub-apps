#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# adb helpers for the generated Android launch.sh.
#
# Functions:
#   select_adb_device
#       Set SERIAL to the connected device, prompting when there are several.
#       Exits if none is connected or the selection is invalid.
#
# Usage: source adb_utils.sh
# ---------------------------------------------------------------------

select_adb_device() {
    local devices=() num_devices=0 serial choice i
    while IFS= read -r serial; do
        [ -n "$serial" ] || continue
        devices+=("$serial")
        num_devices=$((num_devices + 1))
    done < <(adb devices | awk 'NR > 1 && $2 == "device" { print $1 }')

    if [ "$num_devices" -eq 0 ]; then
        echo "::error::No Android device connected. Connect a device with USB debugging enabled." >&2
        exit 1
    elif [ "$num_devices" -eq 1 ]; then
        # shellcheck disable=SC2034
        SERIAL="${devices[0]}"
    else
        echo "Connected devices:"
        for i in "${!devices[@]}"; do
            printf '  %d) %s\n' "$((i + 1))" "${devices[i]}"
        done
        read -rp "Select a device [1-$num_devices]: " choice
        if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "$num_devices" ]; then
            echo "::error::Invalid selection '$choice'." >&2
            exit 1
        fi
        # shellcheck disable=SC2034
        SERIAL="${devices[choice - 1]}"
    fi
}
