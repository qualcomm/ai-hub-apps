# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import shutil
import stat
import subprocess
import sys
import sysconfig

import pytest


def test_run_app(capfd: pytest.CaptureFixture[str]) -> None:
    app_dir = "/qdc/appium/app"

    # Install uv, then use it to provision an isolated venv and install the CLI,
    # mirroring run_linux.sh.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input", "uv"],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    uv_bin = os.path.join(sysconfig.get_path("scripts"), "uv")

    cli_venv = "/data/local/tmp/cli-venv"
    venv_python = os.path.join(cli_venv, "bin", "python3")

    # The CLI is a bundled wheel; its dependencies resolve from PyPI.
    uv_args = ["--pre"]

    test_args = [
        "--app-path",
        app_dir,
        "--device",
        "<<DEVICE_NAME>>",
        "--model-id",
        "<<MODEL_ID>>",
    ]
    if "<<REGISTRY_PATH>>":
        test_args += ["--registry", "<<REGISTRY_PATH>>"]

    # The launch scripts are bash, but the appium host is Alpine (sh/ash only). A static
    # bash is bundled under bin/; make it executable and put it on PATH for the CLI.
    bash_dir = "/qdc/appium/bin"
    bash_bin = os.path.join(bash_dir, "bash")
    os.chmod(
        bash_bin, os.stat(bash_bin).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    env = {
        **os.environ,
        "PATH": bash_dir + os.pathsep + os.environ.get("PATH", ""),
        "QAI_HUB_APPS_EXPERIMENTAL": "1",
        "QAI_HUB_APPS_LOG_LEVEL": "debug",
        # No TTY on the QDC device; skip install-time approval prompts.
        "NON_INTERACTIVE": "true",
    }

    # capfd.disabled(): stream subprocess output live so QDC records it even on pass
    # (pytest otherwise captures fds and only replays them on failure).
    with capfd.disabled():
        print(f"[qdc] bash resolves to: {shutil.which('bash', path=env['PATH'])}")
        print(f"[qdc] adb resolves to: {shutil.which('adb', path=env['PATH'])}")

        subprocess.run(
            [uv_bin, "venv", "--python", sys.executable, cli_venv],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                venv_python,
                *uv_args,
                "<<CLI_SPEC>>",
            ],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        result = subprocess.run(
            [venv_python, "-m", "qai_hub_apps.main", "test", *test_args],
            check=False,
            env=env,
            stdin=subprocess.DEVNULL,
            text=True,
        )

    if result.returncode != 0:
        raise AssertionError("App test failed.")
