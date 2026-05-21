# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import subprocess
import sys


def test_run_instrumentation_test() -> None:
    app_dir = "/qdc/appium/app"
    runner = "<<INSTRUMENTATION_RUNNER>>"

    subprocess.run(
        ["adb", "install", "-r", f"{app_dir}/build/outputs/apk/debug/app-debug.apk"],
        check=True,
    )
    subprocess.run(
        [
            "adb",
            "install",
            "-r",
            f"{app_dir}/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk",
        ],
        check=True,
    )

    result = subprocess.run(
        ["adb", "shell", "am", "instrument", "-w", "-r", runner],
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if (
        result.returncode != 0
        or "INSTRUMENTATION_FAILED" in result.stdout
        or "FAILURES" in result.stdout
    ):
        print(result.stderr, file=sys.stderr)
        raise AssertionError("Instrumentation tests failed.")
