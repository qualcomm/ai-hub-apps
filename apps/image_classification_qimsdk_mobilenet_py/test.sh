#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"
source ../_shared/scripts/pip_utils.sh
activate_venv

TEST_VIDEO="${QAI_HUB_APPS_TEST_VIDEO:-$SCRIPT_DIR/assets/classification_sample_video.mp4}"
if [ ! -f "$TEST_VIDEO" ]; then
    echo "::error::Classification test video not found: $TEST_VIDEO" >&2
    exit 1
fi
OUT_VIDEO="$(mktemp --suffix=.mp4)"
trap 'rm -f "$OUT_VIDEO"' EXIT
echo "Running classification on $TEST_VIDEO"
rc=0
output="$(python main.py \
    --input-type file \
    -i "$TEST_VIDEO" \
    --output file \
    --output-file "$OUT_VIDEO" \
    --log-classifications \
    "$@" 2>&1)" || rc=$?
printf '%s\n' "$output"
if [ "$rc" -ne 0 ]; then
    echo "::error::main.py exited with code $rc." >&2
    exit "$rc"
fi
if [ ! -s "$OUT_VIDEO" ]; then
    echo "::error::Classification output is missing or empty." >&2
    exit 1
fi
gst-discoverer-1.0 "$OUT_VIDEO"

# Check predictions only, not static labels or unrelated SDK messages.
predictions="$(printf '%s\n' "$output" | grep '^\[classification\] ' || true)"
if [ -z "$predictions" ]; then
    echo "::error::No classification predictions found in app output." >&2
    exit 1
fi
for expected in 'monitor|screen|desktop\.computer' 'mouse' 'cup|coffee\.mug'; do
    if ! printf '%s\n' "$predictions" | grep -E "\"label\": \"($expected)\"" >/dev/null; then
        echo "::error::Expected classification label ($expected) not found." >&2
        exit 1
    fi
done
echo "Classification label and MP4 validation passed."
