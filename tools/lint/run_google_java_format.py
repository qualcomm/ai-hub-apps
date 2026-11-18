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
jar = d / "google-java-format.jar"

if not java.exists() or not jar.exists():
    sys.exit("[ERROR] lint toolchain missing. Run tools/setup_env.sh (or tools/setup_env.ps1 on Windows)")

sys.exit(subprocess.run([
    java,
    "--add-exports", "jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED",
    "--add-exports", "jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED",
    "--add-exports", "jdk.compiler/com.sun.tools.javac.parser=ALL-UNNAMED",
    "--add-exports", "jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
    "--add-exports", "jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    "-jar", jar, "--aosp", "--replace", "--set-exit-if-changed", *sys.argv[1:],
]).returncode)
