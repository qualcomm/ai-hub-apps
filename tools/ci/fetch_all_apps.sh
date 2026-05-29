# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Fetch all apps from the registry into a given output directory.
#
# Usage: bash fetch_all_apps.sh <output-dir>

set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output-dir>}"

mkdir -p "$OUTPUT_DIR"

APP_IDS=$(python -c "
from qai_hub_apps.registry import Registry
registry = Registry.load()
print('\n'.join(app.id for app in registry.apps))
")

echo "Apps to fetch:"
echo "$APP_IDS"

while IFS= read -r app_id; do
    echo "--- Fetching $app_id ---"
    for i in {1..2}; do
        qai-hub-apps fetch "$app_id" --output-dir "$OUTPUT_DIR" && break
        echo "Attempt $i failed for $app_id, retrying..."
        sleep 10
        if [ "$i" -eq 2 ]; then
            echo "ERROR: Failed to fetch $app_id after 2 attempts"
            exit 1
        fi
    done
done <<< "$APP_IDS"
