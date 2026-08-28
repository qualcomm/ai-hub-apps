# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from qai_hub_apps import _is_dev
from qai_hub_apps.configs.app_yaml import AppType
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.errors import QAIHubAppsError
from qai_hub_apps.experimental.commands.build import _resolve_app_from_dir, run_build
from qai_hub_apps.experimental.commands.configure import run_configure
from qai_hub_apps.experimental.validate import ensure_run_supported
from qai_hub_apps.registry import App, Registry
from qai_hub_apps.user_config import get_configured_device
from qai_hub_apps.utils.devices import device_env

logger = logging.getLogger(__name__)


def _run_command(
    app: App, app_dir: Path, use_docker: bool, clean: bool, app_args: list[str]
) -> list[str]:
    """Return the command that runs the app's generated launch script."""
    if app.app_type == AppType.WINDOWS:
        script = app_dir / "launch.ps1"
        command = ["powershell", "-File", str(script)]
        no_docker_flag, clean_flag = "-NoDocker", "-Clean"
    else:
        script = app_dir / "launch.sh"
        command = ["bash", str(script)]
        no_docker_flag, clean_flag = "--no-docker", "--clean"
    logger.debug("Launch script for '%s' (%s): %s", app.id, app.app_type.value, script)
    if not script.is_file():
        fix = (
            "Regenerate it with "
            "'python -m qai_hub_apps_test.scripts.generate_app_scripts'."
            if _is_dev()
            else "The app bundle is incomplete; re-fetch it with --overwrite "
            "or pass an updated --app-path, then retry."
        )
        raise QAIHubAppsError(f"No launch script found at '{script}'. {fix}")
    if not use_docker:
        command.append(no_docker_flag)
    if clean:
        command.append(clean_flag)
    if app_args:
        command += ["--", *app_args]
    logger.debug("Run command: %s", command)
    return command


def run_run(
    app_id: str | None,
    app_path: Path | None,
    output_dir: Path,
    registry: Registry,
    model_asset: ModelAsset | None,
    use_docker: bool = True,
    clean: bool = False,
    overwrite: bool = False,
    app_args: list[str] | None = None,
) -> None:
    """Resolve the run target, validate it, build it if needed, and run it."""
    if app_id is not None and app_path is not None:
        raise QAIHubAppsError("Cannot specify both app_id and app_path.")

    logger.debug(
        "run_run: app_id=%s, app_path=%s, use_docker=%s, clean=%s, overwrite=%s, "
        "app_args=%s",
        app_id,
        app_path,
        use_docker,
        clean,
        overwrite,
        app_args,
    )

    device = get_configured_device()
    if device is None:
        logger.info("No target device configured; let's set one up.")
        run_configure(None)
        device = get_configured_device()
    if device is None:
        raise QAIHubAppsError(
            "No target device configured. Run 'qai-hub-apps configure' to select "
            "one before running an app."
        )

    require_build = app_id is not None

    if require_build:
        app = registry.find_by_id(str(app_id))
    else:
        assert app_path is not None
        app_path = app_path.resolve()
        app = _resolve_app_from_dir(app_path, registry)

    ensure_run_supported(app, device, use_docker)

    if require_build:
        if (
            model_asset is None
            and app.related_models
            and not app.disable_cli_model_fetch
        ):
            default_model = app.related_models[0]
            logger.info(
                "No model specified; using '%s' for device '%s'.",
                default_model,
                device.name,
            )
            model_asset = ModelAsset(model_id=default_model, device=device.name)
        app_dir = run_build(
            app_id,
            None,
            output_dir,
            registry,
            model_asset,
            use_docker=use_docker,
            clean=clean,
            overwrite=overwrite,
        )
    else:
        assert app_path is not None
        app_dir = app_path

    device_vars = device_env(device)
    env = {**os.environ, **device_vars}

    command = _run_command(app, app_dir, use_docker, clean, app_args or [])
    logger.info(
        "Running '%s' (%s) on device '%s'...",
        app.id,
        "docker" if use_docker else "native",
        device.name,
    )
    logger.debug("Running %s (cwd=%s)", command, app_dir)
    logger.debug("Using device environment: %s", device_vars)
    try:
        subprocess.run(command, cwd=app_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        raise QAIHubAppsError(
            f"Run failed for '{app.id}' (exit code {e.returncode})."
        ) from e
    logger.debug("Run subprocess for '%s' exited 0", app.id)
    logger.info("Run complete for '%s'.", app.id)
