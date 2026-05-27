#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Parse pytest-json-report output and print either a markdown table or JSON payload.

Usage:
  python build_test_report.py <results_dir> --output-type markdown
  python build_test_report.py <results_dir> --output-type json

Output types:
  markdown  — GitHub-flavored markdown table with per-app/model/stage status
  json      — JSON payload: {"summary": {...}, "results": [...]}
"""

import argparse
import glob
import json
import sys
from pathlib import Path

ICONS = {"passed": "✅", "failed": "❌", "skipped": "⏭"}


def parse_results(results_dir: str) -> dict[tuple[str, str], dict[str, str]]:
    """Parse all pytest-json-report files into a {(app_id, model_id): {stage: outcome}} dict."""
    matrix: dict[tuple[str, str], dict[str, str]] = {}
    for fpath in glob.glob(f"{results_dir}/*.json"):
        data = json.loads(Path(fpath).read_text())
        for test in data.get("tests", []):
            nodeid = test["nodeid"]
            if "test_1" in nodeid:
                stage = "fetch"
            elif "test_2" in nodeid:
                stage = "build"
            else:
                stage = "on_device"
            # nodeid: path::test_N_name[app_id-model_id]
            param = nodeid.split("[")[1].rstrip("]")
            # _param_id format: app_id-model_id
            app_id, model_id = param.rsplit("-", 1)
            matrix.setdefault((app_id, model_id), {})[stage] = test["outcome"]
    return matrix


def build_markdown(matrix: dict[tuple[str, str], dict[str, str]]) -> str:
    lines = [
        "## 📊 App Test Results\n",
        "| App | Model | Fetch | Build | On-Device |",
        "|-----|-------|-------|-------|-----------|",
    ]
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for (app_id, model_id), stages in sorted(matrix.items()):
        fetch = stages.get("fetch", "skipped")
        build = stages.get("build", "skipped")
        on_device = stages.get("on_device", "skipped")
        overall = (
            "failed" if "failed" in (fetch, build, on_device)
            else "passed" if "passed" in (fetch, build, on_device)
            else "skipped"
        )
        counts[overall] += 1
        lines.append(
            f"| {app_id} | {model_id} | {ICONS[fetch]} | {ICONS[build]} | {ICONS[on_device]} |"
        )
    total = sum(counts.values())
    lines.append(
        f"\n**Total:** {total} | "
        f"✅ {counts['passed']} passed | "
        f"❌ {counts['failed']} failed | "
        f"⏭ {counts['skipped']} skipped"
    )
    return "\n".join(lines)


def build_json(matrix: dict[tuple[str, str], dict[str, str]]) -> str:
    rows = []
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for (app_id, model_id), stages in sorted(matrix.items()):
        fetch = stages.get("fetch", "skipped")
        build = stages.get("build", "skipped")
        on_device = stages.get("on_device", "skipped")
        overall = (
            "failed" if "failed" in (fetch, build, on_device)
            else "passed" if "passed" in (fetch, build, on_device)
            else "skipped"
        )
        counts[overall] += 1
        rows.append({
            "app_id": app_id,
            "model_id": model_id,
            "fetch": fetch,
            "build": build,
            "on_device": on_device,
        })
    payload = {"summary": {**counts, "total": sum(counts.values())}, "results": rows}
    return json.dumps(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="Directory containing pytest-json-report *.json files")
    parser.add_argument(
        "--output-type",
        required=True,
        choices=["markdown", "json"],
        help="Output format: markdown table or JSON payload",
    )
    args = parser.parse_args()

    matrix = parse_results(args.results_dir)
    if not matrix:
        print("No test results found.", file=sys.stderr)
        sys.exit(1)

    if args.output_type == "markdown":
        print(build_markdown(matrix))
    else:
        print(build_json(matrix))


if __name__ == "__main__":
    main()
