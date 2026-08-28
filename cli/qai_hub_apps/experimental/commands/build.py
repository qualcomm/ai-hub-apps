# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml

from qai_hub_apps import _is_dev
from qai_hub_apps.commands.fetch import run_fetch
from qai_hub_apps.configs.app_yaml import AppType
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.errors import InvalidArgumentError, QAIHubAppsError
from qai_hub_apps.experimental.validate import ensure_build_supported
from qai_hub_apps.registry import App, Registry

logger = logging.getLogger(__name__)


def _resolve_app_from_dir(app_dir: Path, registry: Registry) -> App:
    """Resolve the registry App for an already-fetched app directory.

    The app id is read from the ``id`` field of the bundled ``info.yaml``.
    """
    info_path = app_dir / "info.yaml"
    logger.debug("Resolving app id from %s", info_path)
    try:
        with open(info_path, encoding="utf-8") as f:
            app_id = yaml.safe_load(f)["id"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as e:
        raise InvalidArgumentError(
            f"Could not read an app 'id' from '{info_path}'. "
            "Pass a directory produced by 'qai-hub-apps fetch'."
        ) from e
    logger.debug("Resolved app id '%s' from %s", app_id, info_path)
    return registry.find_by_id(app_id)


def _prepare_app(
    app_id: str | None,
    app_path: Path | None,
    output_dir: Path,
    registry: Registry,
    model_asset: ModelAsset | None,
    overwrite: bool = False,
) -> tuple[App, Path]:
    """Return the ``(App, app_dir)`` to build, fetching it if needed."""
    assert app_id is None or app_path is None
    logger.debug(
        "prepare_app: app_id=%s, app_path=%s, output_dir=%s, overwrite=%s",
        app_id,
        app_path,
        output_dir,
        overwrite,
    )
    if app_path is not None:
        if model_asset is not None:
            logger.warning(
                "Model options are ignored when building from a path (no fetch)."
            )
        app_dir = app_path.resolve()
        logger.debug("Building in place from path %s", app_dir)
        return _resolve_app_from_dir(app_dir, registry), app_dir

    assert app_id is not None
    app = registry.find_by_id(app_id)
    candidate = output_dir / app.id
    logger.debug("Fetch candidate directory: %s", candidate)
    if candidate.exists() and not overwrite:
        logger.info(
            "Found app at %s, reusing as-is (use --overwrite to re-fetch).",
            candidate.as_posix(),
        )
        if model_asset is not None:
            logger.warning(
                "Model options are ignored when reusing an existing app "
                "directory (no fetch)."
            )
        return app, candidate

    if model_asset is None and not app.disable_cli_model_fetch:
        raise InvalidArgumentError(
            f"Building '{app.id}' requires fetching it, but no model was "
            "provided. Pass --model / --model-id / --model-path, or build an "
            "already-fetched app with --app-path."
        )
    logger.debug("Fetching '%s' into %s (overwrite=%s)", app.id, output_dir, overwrite)
    app_dir = run_fetch(app.id, output_dir, registry, model_asset, overwrite=overwrite)
    return app, app_dir


def _build_command(app: App, app_dir: Path, use_docker: bool, clean: bool) -> list[str]:
    """Return the command that runs the app's generated build script."""
    # Windows apps ship a PowerShell build.ps1; everything else a bash build.sh.
    if app.app_type == AppType.WINDOWS:
        script = app_dir / "build.ps1"
        command = ["powershell", "-File", str(script)]
        no_docker_flag, clean_flag = "-NoDocker", "-Clean"
    else:
        script = app_dir / "build.sh"
        command = ["bash", str(script)]
        no_docker_flag, clean_flag = "--no-docker", "--clean"
    logger.debug("Build script for '%s' (%s): %s", app.id, app.app_type.value, script)
    if not script.is_file():
        fix = (
            "Regenerate it with "
            "'python -m qai_hub_apps_test.scripts.generate_app_scripts'."
            if _is_dev()
            else "The app bundle is incomplete; re-fetch it with --overwrite "
            "or pass an updated --app-path, then retry."
        )
        raise QAIHubAppsError(f"No build script found at '{script}'. {fix}")
    if not use_docker:
        command.append(no_docker_flag)
    if clean:
        command.append(clean_flag)
    logger.debug("Build command: %s", command)
    return command


def run_build(
    app_id: str | None,
    app_path: Path | None,
    output_dir: Path,
    registry: Registry,
    model_asset: ModelAsset | None,
    use_docker: bool = True,
    clean: bool = False,
    overwrite: bool = False,
) -> Path:
    """Resolve the build target, fetch it if needed, and run its build script."""
    logger.debug(
        "run_build: app_id=%s, app_path=%s, use_docker=%s, clean=%s",
        app_id,
        app_path,
        use_docker,
        clean,
    )
    app, app_dir = _prepare_app(
        app_id, app_path, output_dir, registry, model_asset, overwrite=overwrite
    )
    ensure_build_supported(app, use_docker)

    command = _build_command(app, app_dir, use_docker, clean)
    logger.info("Building '%s' (%s)...", app.id, "docker" if use_docker else "native")
    logger.debug("Running %s (cwd=%s)", command, app_dir)
    try:
        subprocess.run(command, cwd=app_dir, check=True)
    except subprocess.CalledProcessError as e:
        raise QAIHubAppsError(
            f"Build failed for '{app.id}' (exit code {e.returncode})."
        ) from e
    logger.debug("Build subprocess for '%s' exited 0", app.id)
    logger.info("Build complete for '%s'.", app.id)
    return app_dir
