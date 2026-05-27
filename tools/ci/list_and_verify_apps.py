#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Group CLI-registered apps by app_type, verify full coverage, and print results.

Prints one KEY=JSON_ARRAY line per app type to stdout:
  ubuntu=["app_a", "app_b"]
  android=["app_c"]
  windows=[]

Exits non-zero if any registry app is not covered by any group.

Usage:
  python list_and_verify_apps.py
"""

import json
import sys

from qai_hub_apps.registry import Registry
from qai_hub_apps_test.configs.info_yaml import QAIHAAppInfo

groups: dict[str, list[str]] = {"ubuntu": [], "android": [], "windows": []}
registry = Registry.load()

for app in registry.apps:
    info, _ = QAIHAAppInfo.from_app(app.id)
    groups[info.app_type.value].append(app.id)


all_apps = {app.id for app in registry.apps}
covered = set(groups["ubuntu"] + groups["android"] + groups["windows"])
missing = all_apps - covered
if missing:
    print(f"ERROR: Apps not covered by any group: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)

print(
    f"Coverage OK: {len(all_apps)} apps — "
    f"ubuntu={len(groups['ubuntu'])}, "
    f"android={len(groups['android'])}, "
    f"windows={len(groups['windows'])}",
    file=sys.stderr,
)

for key, ids in groups.items():
    print(f"{key}={json.dumps(ids)}")
