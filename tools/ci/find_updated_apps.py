#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Find registered app IDs that are affected by a set of changed file paths.

Reads a diff file containing one repo-relative file path per line (i.e. the
output of `git diff --name-only`) and determines which registered apps need
to be rebuilt. Two detection strategies are combined:

1. Direct changes — a file under apps/<app_id>/ maps directly to that app.
2. Bundled changes — a file under apps/_shared/ or tools/docker/
   is SHA-256 hashed; all registered apps are fetched and their bundles are
   walked file-by-file. Any app whose bundle contains a file with a matching
   hash is affected.

Outputs GITHUB_OUTPUT-compatible lines:

  has_app_changes=true
  app_filter=chatapp_android,image_classification_android

Usage:
  git diff --name-only main > changed.txt
  python find_updated_apps.py --diff-file changed.txt
"""

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from qai_hub_apps.registry import Registry
from qai_hub_apps_test.utils.paths import (
    APPS_ROOT,
    DOCKER_ROOT,
    REPOSITORY_ROOT,
    SHARED_UTILS_ROOT,
)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def find_affected_by_bundled(bundled_files: list[str], app_ids: list[str]) -> set[str]:
    """Return the IDs of registered apps whose bundle contains any of the changed files."""
    bundled_hashes: set[str] = set()
    for rel_path in bundled_files:
        abs_path = REPOSITORY_ROOT / rel_path
        if abs_path.is_file():
            bundled_hashes.add(file_hash(abs_path))
        else:
            print(f"Warning: bundled file not found on disk: {abs_path}", file=sys.stderr)

    if not bundled_hashes:
        return set()

    affected: set[str] = set()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for app_id in app_ids:
            app_fetch_dir = tmp_path / app_id
            try:
                subprocess.run(
                    ["qai-hub-apps", "fetch", app_id, "--output-dir", str(tmp_path)],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Warning: failed to fetch {app_id}: {e.stderr.decode()}", file=sys.stderr)
                continue

            for fetched_file in app_fetch_dir.rglob("*"):
                if fetched_file.is_file() and file_hash(fetched_file) in bundled_hashes:
                    print(f"  {app_id}: matched {fetched_file.relative_to(app_fetch_dir)}", file=sys.stderr)
                    affected.add(app_id)
                    break  # one match is enough, move to next app

    return affected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--diff-file",
        required=True,
        type=Path,
        help="Path to a file containing one repo-relative changed file path per line (git diff --name-only output)",
    )
    args = parser.parse_args()

    diff_files = [line for line in args.diff_file.read_text().splitlines() if line.strip()]

    if not diff_files:
        print("has_app_changes=false")
        print("app_filter=")
        return

    registry = Registry.load()
    app_ids = [app.id for app in registry.apps]

    docker_prefix = DOCKER_ROOT.relative_to(REPOSITORY_ROOT).parts

    direct_changes: set[str] = set()
    bundled_changes: list[str] = []

    for path in diff_files:
        parts = Path(path).parts
        # Check if the file is a Dockerfile under DOCKER_ROOT
        if parts[: len(docker_prefix)] == docker_prefix:
            bundled_changes.append(path)
            continue
        # Skip if the file is not under the APPS_ROOT
        if len(parts) < 2 or parts[0] != APPS_ROOT.name:
            continue
        # Take if the file is under SHARED_UTILS_ROOT or directly under the app
        if parts[1] == SHARED_UTILS_ROOT.name:
            bundled_changes.append(path)
        elif parts[1] in app_ids:
            direct_changes.add(parts[1])

    print(f"Direct app changes: {sorted(direct_changes)}", file=sys.stderr)
    print(f"Bundled file changes: {bundled_changes}", file=sys.stderr)

    bundled_affected = (
        find_affected_by_bundled(bundled_changes, app_ids) if bundled_changes else set()
    )

    matched = sorted(direct_changes | bundled_affected)

    if matched:
        print("has_app_changes=true")
        print(f"app_filter={','.join(matched)}")
        print(f"Matched registered apps: {matched}", file=sys.stderr)
    else:
        print("has_app_changes=false")
        print("app_filter=")
        print("No registered apps affected.", file=sys.stderr)


if __name__ == "__main__":
    main()
