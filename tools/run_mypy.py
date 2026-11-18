#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import subprocess, sys
from pathlib import Path

repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
_ext = ".exe" if sys.platform == "win32" else ""
mypy = repo / "qaiha-dev" / ("Scripts" if sys.platform == "win32" else "bin") / f"mypy{_ext}"

if not mypy.exists():
    sys.exit("[ERROR] qaiha-dev not found. Run tools/setup_env.sh (or tools/setup_env.ps1 on Windows)")

if sys.platform != "linux":
    print(
        f"[WARNING] mypy results on {sys.platform} may not match Linux, which CI uses for PR acceptance. "
        "Use Linux or WSL to reproduce CI results. See CONTRIBUTING.md#mypy-results-differ-from-ci.",
        file=sys.stderr,
    )

sys.exit(subprocess.run([mypy, *sys.argv[2:]], cwd=repo / sys.argv[1]).returncode)
