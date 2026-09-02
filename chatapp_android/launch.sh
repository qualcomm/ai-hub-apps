#!/usr/bin/env bash

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

PACKAGE="com.quicinc.chatapp"
APK="build/outputs/apk/debug/app-debug.apk"
TEST_APK="build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
RUNNER="com.quicinc.chatapp.test/androidx.test.runner.AndroidJUnitRunner"

RUN_TEST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --test) RUN_TEST=1 ;;
        --no-docker|--docker|--clean) ;;
        --) shift; break ;;
        *) echo "::error::Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

if ! command -v adb >/dev/null 2>&1; then
    echo "::error::adb not found on PATH. Install the Android platform-tools." >&2
    exit 1
fi

if [ ! -f "$APK" ]; then
    echo "::error::APK not found at $APK for chatapp_android. Build it first." >&2
    exit 1
fi
if [ "$RUN_TEST" -eq 1 ] && [ ! -f "$TEST_APK" ]; then
    echo "::error::Test APK not found at $TEST_APK for chatapp_android. Build it first." >&2
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

if [ "$RUN_TEST" -eq 1 ]; then
    echo "::step::Installing $TEST_APK on $SERIAL"
    adb -s "$SERIAL" install -r -t "$TEST_APK"

    echo "::step::Running instrumentation tests for chatapp_android"
    set +e
    OUTPUT="$(adb -s "$SERIAL" shell am instrument -w -r "$RUNNER" 2>&1)"
    RC=$?
    set -e
    echo "$OUTPUT"
    if [ "$RC" -ne 0 ] || echo "$OUTPUT" | grep -qE "INSTRUMENTATION_FAILED|FAILURES"; then
        echo "::error::Instrumentation tests failed for chatapp_android." >&2
        exit 1
    fi
    echo "::done::test"
else
    echo "::step::Launching chatapp_android ($PACKAGE)"
    adb -s "$SERIAL" shell monkey -p "$PACKAGE" -c android.intent.category.LAUNCHER 1
    echo "::done::run"
fi
