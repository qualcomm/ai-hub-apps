#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Find registered app IDs whose fetched bundle differs between two git refs.

Each ref is checked out into a worktree, its own CLI + tooling is installed, and
every registered app is fetched. An app is affected if its bundle contents differ
between the two fetches, or if it exists at only one ref.

Outputs GITHUB_OUTPUT-compatible lines:

  has_app_changes=true
  app_filter=chatapp_android,image_classification_android

Usage:
  python find_updated_apps.py --base-ref origin/main
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

# Regenerated per fetch (timestamp, git-derived cli_version), so never a real change.
IGNORED = ["qai_hub_apps.json", "__pycache__"]

logger = logging.getLogger(__name__)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if p.returncode:
        sys.exit(f"Command failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p


def fetch_all_apps(worktree: Path, out_dir: Path) -> set[str]:
    """Stage the worktree's apps via its own script, and return their ids."""
    logger.info("=> staging apps")
    run(
        [
            "bash",
            "tools/ci/stage_all_apps.sh",
            f"--venv={worktree / '.venv_detect'}",
            f"--outdir={out_dir}",
        ],
        cwd=worktree,
    )
    app_ids = {d.name for d in out_dir.iterdir() if d.is_dir()}
    if not app_ids:
        sys.exit(f"No apps were fetched into {out_dir}")
    return app_ids


def diff_apps(base: Path, head: Path) -> str:
    """Return a unified diff of two fetched app directories, empty when identical."""
    cmd = ["diff", "-ru"]
    for name in IGNORED:
        cmd += ["-x", name]
    # Run from the shared parent so the diff names bundle-relative paths, not tmp dirs.
    root = base.parent.parent
    return subprocess.run(
        [*cmd, str(base.relative_to(root)), str(head.relative_to(root))],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-ref", required=True, help="Git ref to compare against")
    parser.add_argument("--head-ref", default="HEAD", help="Git ref under test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sides: dict[str, tuple[Path, set[str]]] = {}
        worktrees: list[Path] = []
        try:
            for side, ref in (("base", args.base_ref), ("head", args.head_ref)):
                logger.info(f"=> resolving {side} ref")
                sha = run(
                    ["git", "rev-parse", "--short", ref],
                ).stdout.strip()
                logger.info("%s ref: %s (%s)", side, ref, sha)
                worktree = tmp_path / f"wt_{side}"
                logger.info("=> adding worktree")
                run(
                    ["git", "worktree", "add", "-q", "--detach", str(worktree), ref],
                )
                worktrees.append(worktree)
                out_dir = tmp_path / f"out_{side}"
                out_dir.mkdir()
                sides[side] = (out_dir, fetch_all_apps(worktree, out_dir))

            base_dir, base_ids = sides["base"]
            head_dir, head_ids = sides["head"]

            matched: list[str] = []
            for app_id in sorted(head_ids):
                if app_id not in base_ids:
                    matched.append(app_id)
                    logger.info("%s: newly registered at head", app_id)
                elif diff := diff_apps(base_dir / app_id, head_dir / app_id):
                    matched.append(app_id)
                    logger.info("%s:\n%s", app_id, diff)
        finally:
            # --force: each worktree holds an untracked venv and a regenerated registry.
            for worktree in worktrees:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    check=False,
                    capture_output=True,
                )

    if matched:
        print("has_app_changes=true")
        print(f"app_filter={','.join(matched)}")
    else:
        logger.info("no apps affected")
        print("has_app_changes=false")
        print("app_filter=")


if __name__ == "__main__":
    main()
