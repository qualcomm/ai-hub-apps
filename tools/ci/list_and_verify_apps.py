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
  python list_and_verify_apps.py --only chatapp_android,image_classification_android  # no coverage check
"""

import argparse
import json
import sys

from qai_hub_apps.registry import Registry
from qai_hub_apps_test.configs.info_yaml import QAIHAAppInfo

parser = argparse.ArgumentParser()
parser.add_argument(
    "--only",
    default=set(),
    type=lambda s: {a.strip() for a in s.split(",") if a.strip()},
    help="Comma-separated app IDs to include (e.g. chatapp_android,image_classification_android). "
    "Omit to include all registered apps.",
)
args = parser.parse_args()

groups: dict[str, list[str]] = {"ubuntu": [], "android": [], "windows": []}
registry = Registry.load()
all_apps = set()
for app in registry.apps:
    all_apps.add(app.id)
    if args.only and app.id not in args.only:
        continue
    info, _ = QAIHAAppInfo.from_app(app.id)
    groups[info.app_type.value].append(app.id)

if args.only:
    unknown = args.only - all_apps
    if unknown:
        print(f"ERROR: Unknown app IDs: {sorted(unknown)}", file=sys.stderr)
        sys.exit(1)
else:
    covered = set(groups["ubuntu"] + groups["android"] + groups["windows"])
    missing = all_apps - covered
    if missing:
        print(f"ERROR: Apps not covered by any group: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

print(
    f"Coverage OK: {sum(len(v) for v in groups.values())} apps {f'(filtered to: {sorted(args.only)}) ' if args.only else ''}— "
    f"ubuntu={len(groups['ubuntu'])}, "
    f"android={len(groups['android'])}, "
    f"windows={len(groups['windows'])}",
    file=sys.stderr,
)

for key, ids in groups.items():
    print(f"{key}={json.dumps(ids)}")
