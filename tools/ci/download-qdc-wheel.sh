# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# Downloads the QDC SDK zip, verifies its checksum, extracts the wheel,
# copies it to the repo root, and cleans up temporary files.
#
# Usage: download-qdc-wheel.sh <destination_dir>

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

QDC_SDK_URL="https://softwarecenter.qualcomm.com/api/download/software/tools/Qualcomm_Device_Cloud_SDK/All/0.4.1/qualcomm_device_cloud_sdk-0.4.1.zip"
QDC_SDK_SHA256="716a862ce64f9146078cd0b7b7ab18d2672520e068345accbb094e848cc22cfb"
DEST_DIR="$1"

TMP_ZIP=/tmp/qualcomm_device_cloud_sdk.zip
TMP_DIR=/tmp/qualcomm_device_cloud_sdk

download_and_verify "$QDC_SDK_URL" "$TMP_ZIP" "$QDC_SDK_SHA256"
unzip -q "$TMP_ZIP" -d "$TMP_DIR"

wheels=("$TMP_DIR"/*.whl)
[ "${#wheels[@]}" -eq 1 ] || { echo "Expected exactly one .whl, found ${#wheels[@]}"; exit 1; }
cp "${wheels[0]}" "$DEST_DIR/"

rm -rf "$TMP_DIR" "$TMP_ZIP"
