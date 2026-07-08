#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail

# shellcheck source=tools/lint/lint_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lint_common.sh"

CHECKSTYLE_JAR="$TOOLS_DIR/checkstyle.jar"

if [ ! -x "$JAVA" ] || [ ! -f "$CHECKSTYLE_JAR" ]; then
    echo "[ERROR] lint toolchain missing. Run 'bash tools/setup_env.sh --extras precommit'" >&2
    exit 1
fi

exec "$JAVA" -jar "$CHECKSTYLE_JAR" -c "$(repo_root)/.checkstyle.xml" "$@"
