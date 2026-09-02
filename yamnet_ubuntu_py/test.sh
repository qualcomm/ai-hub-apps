#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/qairt_utils.sh"

TEST_ASSET_BASE="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-apps/apps/yamnet_ubuntu_py/test"

# Each record is "<wav filename>|<class expected in the top-5 predictions>".
# The expected class is matched case-insensitively against the app's output.
TESTS=(
    "speech_whistling2.wav|Whistling"
    "bird_cawing.wav|Bird"
    "tune.wav|Music"
)

if [ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    echo "error: virtual environment not found. Run install_runtime.sh first." >&2
    exit 1
fi
source "$SCRIPT_DIR/.venv/bin/activate"

failures=0
for test in "${TESTS[@]}"; do
    wav_name="${test%%|*}"
    expected="${test##*|}"
    wav_path="$SCRIPT_DIR/$wav_name"

    wget -q -O "$wav_path" "$TEST_ASSET_BASE/$wav_name"

    echo "=== $wav_name (expecting '$expected') ==="
    output="$(python main.py --audio-file "$wav_path" --qairt-path "$QAIRT_PATH")"
    echo "$output"

    if echo "$output" | grep -qi -- "$expected"; then
        echo "PASS: '$expected' found in predictions"
    else
        echo "FAIL: '$expected' not found in predictions" >&2
        failures=$((failures + 1))
    fi
done

if [ "$failures" -ne 0 ]; then
    echo "$failures test case(s) failed." >&2
    exit 1
fi
echo "All test cases passed."
