# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Build the qai-hub-apps CLI wheel.

Run from the repository root; prints the built wheel's absolute path.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the qai-hub-apps CLI wheel.")
    parser.add_argument("out_dir", help="Directory to write the wheel to.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for whl in glob.glob(os.path.join(args.out_dir, "*.whl")):
        os.remove(whl)
    for egg in glob.glob(os.path.join("cli", "**", "*.egg-info"), recursive=True):
        shutil.rmtree(egg, ignore_errors=True)

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "build"], stdout=sys.stderr
    )
    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", "--outdir", args.out_dir, "cli"],
        stdout=sys.stderr,
    )

    print(os.path.abspath(glob.glob(os.path.join(args.out_dir, "*.whl"))[0]))


if __name__ == "__main__":
    main()
