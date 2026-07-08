#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# shellcheck source=tools/ci/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../ci/common.sh"

TOOLS_DIR="$(repo_root)/.lint-tools"
# shellcheck disable=SC2034
JAVA="$TOOLS_DIR/jdk/bin/java"
