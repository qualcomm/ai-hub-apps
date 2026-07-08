#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail

# shellcheck source=tools/lint/lint_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lint_common.sh"

GJF_JAR="$TOOLS_DIR/google-java-format.jar"

if [ ! -x "$JAVA" ] || [ ! -f "$GJF_JAR" ]; then
    echo "[ERROR] lint toolchain missing. Run 'bash tools/setup_env.sh --extras precommit'" >&2
    exit 1
fi

exec "$JAVA" \
    --add-exports jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED \
    --add-exports jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED \
    --add-exports jdk.compiler/com.sun.tools.javac.parser=ALL-UNNAMED \
    --add-exports jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED \
    --add-exports jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED \
    -jar "$GJF_JAR" --aosp --replace --set-exit-if-changed "$@"
