#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Resolve the release/nightly version and print it for a workflow to consume.

Prints GITHUB_OUTPUT-compatible key=value lines to stdout (redirect with
`>> $GITHUB_OUTPUT`).

  version=0.32.3.dev3+g0c576490c
  is_dev=true
  sha=0c57649

Version source follows the release target:
  dev            setuptools_scm derives it from the commit (dev builds)
  staging/prod   the git tag (minus a leading 'v') is the version; the run must
                 be from a tag (read from $GITHUB_REF / $GITHUB_REF_NAME)

With --expected, fail unless the resolved version matches it.

Usage:
  python tools/ci/resolve_version.py --target dev >> "$GITHUB_OUTPUT"
  python tools/ci/resolve_version.py --target prod \\
      --expected "$EXPECTED_VERSION" >> "$GITHUB_OUTPUT"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from setuptools_scm import get_version

from qai_hub_apps_test.utils.paths import REPOSITORY_ROOT
from qai_hub_apps_test.utils.versions import is_dev


def resolve_version(target: str) -> str:
    """Return the version string for the given release target.

    A ``dev`` target derives the version from the commit via setuptools_scm;
    any other target reads it from the git tag (minus a leading 'v'), taken
    from $GITHUB_REF / $GITHUB_REF_NAME.
    """
    if target == "dev":
        return get_version(root=str(REPOSITORY_ROOT))
    # Non-dev targets must run from a tag; the tag (minus 'v') is the version.
    try:
        ref = os.environ["GITHUB_REF"]
        ref_name = os.environ["GITHUB_REF_NAME"]
    except KeyError as missing:
        sys.exit(f"Error: {missing} must be set for target '{target}'.")
    if not ref.startswith("refs/tags/"):
        sys.exit(f"Error: target '{target}' must run from a tag, got '{ref}'.")
    return ref_name[1:] if ref_name.startswith("v") else ref_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("dev", "staging", "prod"))
    parser.add_argument(
        "--expected", default="", help="fail unless the resolved version matches this"
    )
    args = parser.parse_args()

    version = resolve_version(args.target)

    if args.expected and version != args.expected:
        sys.exit(f"Error: resolved version '{version}' != expected '{args.expected}'.")

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    print(
        f"Resolved version: {version} (is_dev={is_dev(version)}, sha={sha})",
        file=sys.stderr,
    )
    print(f"version={version}")
    print(f"is_dev={'true' if is_dev(version) else 'false'}")
    print(f"sha={sha}")


if __name__ == "__main__":
    main()
