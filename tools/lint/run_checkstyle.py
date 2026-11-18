#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import subprocess, sys
from pathlib import Path

repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
d = repo / ".lint-tools"
java = d / "jdk" / "bin" / ("java.exe" if sys.platform == "win32" else "java")
jar = d / "checkstyle.jar"

if not java.exists() or not jar.exists():
    sys.exit("[ERROR] lint toolchain missing. Run tools/setup_env.sh (or tools/setup_env.ps1 on Windows)")

sys.exit(subprocess.run([java, "-jar", jar, "-c", repo / ".checkstyle.xml", *sys.argv[1:]]).returncode)
