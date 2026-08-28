#!/usr/bin/env bash

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

APK="build/outputs/apk/debug/app-debug.apk"
PACKAGE="com.quicinc.chatapp"

if ! command -v adb >/dev/null 2>&1; then
    echo "::error::adb not found on PATH. Install the Android platform-tools." >&2
    exit 1
fi

if [ ! -f "$APK" ]; then
    echo "::error::APK not found at $APK for chatapp_android. Build it first." >&2
    exit 1
fi

DEVICES=()
NUM_DEVICES=0
while IFS= read -r serial; do
    [ -n "$serial" ] || continue
    DEVICES+=("$serial")
    NUM_DEVICES=$((NUM_DEVICES + 1))
done < <(adb devices | awk 'NR > 1 && $2 == "device" { print $1 }')

if [ "$NUM_DEVICES" -eq 0 ]; then
    echo "::error::No Android device connected. Connect a device with USB debugging enabled." >&2
    exit 1
elif [ "$NUM_DEVICES" -eq 1 ]; then
    SERIAL="${DEVICES[0]}"
else
    echo "Connected devices:"
    for i in "${!DEVICES[@]}"; do
        printf '  %d) %s\n' "$((i + 1))" "${DEVICES[i]}"
    done
    read -rp "Select a device [1-$NUM_DEVICES]: " choice
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "$NUM_DEVICES" ]; then
        echo "::error::Invalid selection '$choice'." >&2
        exit 1
    fi
    SERIAL="${DEVICES[choice - 1]}"
fi

echo "::step::Installing $APK on $SERIAL"
adb -s "$SERIAL" install -r -t "$APK"

echo "::step::Launching chatapp_android ($PACKAGE)"
adb -s "$SERIAL" shell monkey -p "$PACKAGE" -c android.intent.category.LAUNCHER 1
echo "::done::run"
