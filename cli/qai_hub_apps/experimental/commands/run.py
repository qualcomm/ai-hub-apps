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
from qai_hub_apps.experimental.commands.configure import (
    prompt_for_device,
    run_configure,
)
from qai_hub_apps.experimental.validate import ensure_run_supported
from qai_hub_apps.registry import App, Registry
from qai_hub_apps.user_config import get_configured_device
from qai_hub_apps.utils.devices import (
    device_env,
    list_android_devices,
    resolve_device_info,
)

logger = logging.getLogger(__name__)


def _run_command(
    app: App,
    app_dir: Path,
    use_docker: bool,
    clean: bool,
    app_args: list[str],
    test: bool,
) -> list[str]:
    """Return the command that runs the app's generated launch script."""
    if app.app_type == AppType.WINDOWS:
        script = app_dir / "launch.ps1"
        command = ["powershell", "-File", str(script)]
        no_docker_flag, clean_flag = "-NoDocker", "-Clean"
        test_flag = "-Test"
    else:
        script = app_dir / "launch.sh"
        command = ["bash", str(script)]
        no_docker_flag, clean_flag = "--no-docker", "--clean"
        test_flag = "--test"
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
    if test:
        command.append(test_flag)
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
    test: bool = False,
) -> None:
    """Resolve the run target, validate it, build it if needed, and run it."""
    if app_id is not None and app_path is not None:
        raise QAIHubAppsError("Cannot specify both app_id and app_path.")

    logger.debug(
        "run_run: app_id=%s, app_path=%s, use_docker=%s, clean=%s, overwrite=%s, "
        "app_args=%s, test=%s",
        app_id,
        app_path,
        use_docker,
        clean,
        overwrite,
        app_args,
        test,
    )

    require_build = app_id is not None

    if require_build:
        app = registry.find_by_id(str(app_id))
    else:
        assert app_path is not None
        app_path = app_path.resolve()
        app = _resolve_app_from_dir(app_path, registry)
        if model_asset is not None:
            logger.warning(
                "Running an already-fetched app; --model/--model-id are not "
                "used. Using device '%s' as the run target.",
                model_asset.device or "<configured>",
            )

    # Windows apps can be built in a Windows container, but always run natively.
    run_docker = use_docker
    if app.app_type == AppType.WINDOWS and run_docker:
        logger.info("Windows apps run natively; ignoring Docker mode for the run.")
        run_docker = False

    override = (
        resolve_device_info(model_asset.device)
        if model_asset is not None and model_asset.device
        else None
    )
    # Android apps run on a mobile device, not the configured environment device,
    # so pick an Android target for this run.
    if override is not None:
        device = override
    elif app.app_type == AppType.ANDROID:
        android_devices = list_android_devices()
        if app.supported_devices:
            android_devices = [
                d for d in android_devices if d.name in app.supported_devices
            ]
        device = prompt_for_device(android_devices, title="Select your Android device")
    else:
        device = get_configured_device()
        if device is None:
            logger.info("No target device configured; let's set one up.")
            run_configure(None)
            device = get_configured_device()
        if device is None:
            raise QAIHubAppsError(
                "No target device configured. Run 'qai-hub-apps configure' to "
                "select one before running an app."
            )

    ensure_run_supported(app, device, run_docker)

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

    command = _run_command(app, app_dir, run_docker, clean, app_args or [], test)
    logger.info(
        "Running '%s' (%s) on device '%s'...",
        app.id,
        "docker" if run_docker else "native",
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
