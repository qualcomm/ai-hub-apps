# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Install this checkout's tooling, generate the test-scope registry, and fetch
# every registered app into --outdir.
#
# find_updated_apps.py runs each git ref's own copy of this script, so --venv and
# --outdir are a cross-ref contract: keep them on every branch.
#
# Usage: bash stage_all_apps.sh --venv <path> --outdir <path>

set -euo pipefail

# shellcheck source=tools/ci/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

VENV=""
OUTDIR=""

while [ $# -gt 0 ]; do
    case $1 in
        --venv) VENV="$2"; shift ;;
        --venv=*) VENV="${1#--venv=}" ;;
        --outdir) OUTDIR="$2"; shift ;;
        --outdir=*) OUTDIR="${1#--outdir=}" ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

[ -n "$VENV" ] || { echo "error: --venv is required" >&2; exit 1; }
[ -n "$OUTDIR" ] || { echo "error: --outdir is required" >&2; exit 1; }

cd "$(repo_root)"

bash tools/setup_env.sh --venv="$VENV"
bash tools/setup_env.sh --venv="$VENV" --with-cli
"$VENV/bin/python" -m qai_hub_apps_test.scripts.generate_registry \
    --output_dir cli/qai_hub_apps/ --scope test

# fetch_all_apps.sh calls `python` and `qai-hub-apps`.
PATH="$VENV/bin:$PATH" bash tools/ci/fetch_all_apps.sh "$OUTDIR"
