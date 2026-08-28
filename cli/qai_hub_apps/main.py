# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import sys
from pathlib import Path

from qai_hub_apps import PACKAGE_NAME, __version__, _is_dev
from qai_hub_apps.commands.fetch import run_fetch
from qai_hub_apps.commands.list_apps import run_info, run_list
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.errors import (
    InvalidArgumentError,
    QAIHubAppsError,
    RegistryNotFoundError,
)
from qai_hub_apps.experimental import add_experimental_parser
from qai_hub_apps.experimental.commands.build import run_build
from qai_hub_apps.experimental.commands.configure import run_configure
from qai_hub_apps.experimental.commands.run import run_run
from qai_hub_apps.logging_utils import configure_logging
from qai_hub_apps.registry import Registry
from qai_hub_apps.utils.updates import check_for_update

logger = logging.getLogger(__name__)


def _resolve_model_asset(
    model: str | None,
    model_id: str | None,
    model_path: Path | None,
    chipset: str | None,
    device: str | None,
) -> ModelAsset | None:
    """Build a ModelAsset from the --model / --model-id / --model-path args.

    --model-id and --model-path are explicit. Plain --model is auto-resolved: an
    existing path is treated as a local export, otherwise as a model ID. (A
    --model value that is both a supported model and a path is rejected later,
    in App.fetch, with a hint to use the explicit flags.)

    --chipset and --device are mutually exclusive (enforced by argparse).
    """
    if model_id is not None:
        return ModelAsset(model_id=model_id, chipset=chipset, device=device)
    if model_path is not None:
        # Resolve to an absolute path so it can never collide with a model id.
        return ModelAsset(path=model_path.resolve())
    if model is None:
        return None
    path = Path(model)
    if path.exists():
        if chipset is not None or device is not None:
            flag = "--device" if device is not None else "--chipset"
            logger.warning("%s is ignored when --model is a local path.", flag)
        return ModelAsset(path=path)
    return ModelAsset(model_id=model, chipset=chipset, device=device)


def _resolve_app_target(
    app: str | None,
    app_id: str | None,
    app_path: Path | None,
    overwrite: bool,
) -> tuple[str | None, Path | None]:
    """Resolve the build target into an (app_id, app_path) pair.

    --app-id and --app-path are explicit. The positional ``app`` is auto-resolved:
    an existing directory is treated as a fetched app path, otherwise as an app
    ID.
    """
    if app is None:
        return app_id, app_path
    if Path(app).exists():
        if overwrite:
            raise InvalidArgumentError(
                f"--overwrite has no effect when building an existing directory: "
                f"'{app}' is a local path ({Path(app).resolve()}), which is built in place. "
                f"Drop --overwrite to build it as-is, or pass '--app-id {app}' "
                f"to fetch the app (overwriting any existing copy) and build it."
            )
        logger.debug("Auto-resolved '%s' to existing path", app)
        return None, Path(app)
    logger.debug("Auto-resolved '%s' to app id", app)
    return app, None


def _split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv at the first ``--``; the tail is passed through to the run target."""
    if "--" in argv:
        sep = argv.index("--")
        return argv[:sep], argv[sep + 1 :]
    return argv, []


def main() -> None:
    epilog = (
        "Examples:\n"
        "  qai-hub-apps list                   List all available apps\n"
        "  qai-hub-apps info <app_id>          Show details for an app\n"
        "  qai-hub-apps fetch <app_id>         Download an app's source\n"
    )
    parser = argparse.ArgumentParser(
        prog=PACKAGE_NAME,
        description="CLI for browsing and downloading Qualcomm® AI Hub Apps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log-level",
        dest="log_level",
        choices=["debug", "info", "error"],
        default=None,
        help="Logging verbosity (overrides QAI_HUB_APPS_LOG_LEVEL; default: info)",
    )
    log_group.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        action="store_const",
        const="debug",
        help="Show debug-level diagnostics (same as --log-level debug)",
    )
    log_group.add_argument(
        "-q",
        "--quiet",
        dest="log_level",
        action="store_const",
        const="error",
        help="Only show errors (same as --log-level error)",
    )

    subparsers = parser.add_subparsers(dest="command")

    def add_registry_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--registry",
            type=Path,
            default=None,
            help="Path to registry.yaml (defaults to bundled registry)"
            if _is_dev()
            else argparse.SUPPRESS,
        )

    def add_app_id_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "app_id",
            help="App ID (from 'qai-hub-apps list')",
        )

    def add_fetch_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "-o",
            "--output-dir",
            dest="output_dir",
            type=Path,
            default=Path.cwd(),
            help="Output directory (default: current directory)",
        )
        model_group = p.add_mutually_exclusive_group()
        model_group.add_argument(
            "--model",
            dest="model",
            default=None,
            metavar="MODEL_ID_OR_PATH",
            help="Model to bundle: a model ID to download (must be supported by the app), "
            "or a path to a locally-exported model (directory or .zip). "
            "Use --model-id or --model-path to be explicit",
        )
        model_group.add_argument(
            "--model-id",
            dest="model_id",
            default=None,
            metavar="MODEL_ID",
            help="Model ID to download (must be supported by the app)",
        )
        model_group.add_argument(
            "--model-path",
            dest="model_path",
            default=None,
            type=Path,
            metavar="PATH",
            help="Path to a locally-exported model (directory or .zip)",
        )
        target_group = p.add_mutually_exclusive_group()
        target_group.add_argument(
            "--chipset",
            dest="chipset",
            default=None,
            metavar="CHIPSET",
            help="Chipset to target when downloading model (must be supported by the app)",
        )
        target_group.add_argument(
            "--device",
            dest="device",
            default=None,
            metavar="DEVICE",
            help="Device to target when downloading model (must be supported by the app)",
        )
        p.add_argument(
            "--overwrite",
            dest="overwrite",
            action="store_true",
            help="Overwrite the app in place if it already exists in the output "
            "directory (default: save a separate numbered copy)",
        )

    def add_app_action_args(p: argparse.ArgumentParser, verb: str) -> None:
        """Add the shared app-target + fetch + docker/clean args to build/run."""
        add_registry_arg(p)
        app_group = p.add_mutually_exclusive_group()
        app_group.add_argument(
            "app",
            nargs="?",
            default=None,
            metavar="APP_ID_OR_PATH",
            help=f"App to {verb}: an app ID to fetch-and-{verb} (from "
            "'qai-hub-apps list'), or a path to an already-fetched app directory. "
            "Use --app-id or --app-path to be explicit",
        )
        app_group.add_argument(
            "--app-id",
            dest="app_id",
            default=None,
            metavar="APP_ID",
            help=f"App ID to {verb} (from 'qai-hub-apps list'); fetched first if "
            "needed",
        )
        app_group.add_argument(
            "--app-path",
            dest="app_path",
            default=None,
            type=Path,
            metavar="PATH",
            help=f"Path to an already-fetched app directory to {verb} in place",
        )
        add_fetch_args(p)
        p.add_argument(
            "--no-docker",
            dest="no_docker",
            action="store_true",
            help=f"{verb.capitalize()} natively on the host instead of using Docker",
        )
        p.add_argument(
            "--clean",
            dest="clean",
            action="store_true",
            help="Cleanup prior build artifacts before building",
        )

    list_parser = subparsers.add_parser("list", help="List available apps")
    add_registry_arg(list_parser)

    info_parser = subparsers.add_parser("info", help="Show details for an app")
    add_registry_arg(info_parser)
    add_app_id_arg(info_parser)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Download and extract an app's source"
    )
    add_registry_arg(fetch_parser)
    add_app_id_arg(fetch_parser)
    add_fetch_args(fetch_parser)

    build_parser = add_experimental_parser(subparsers, "build", help="Build an app")
    add_app_action_args(build_parser, "build")

    run_parser = add_experimental_parser(subparsers, "run", help="Run an app")
    add_app_action_args(run_parser, "run")

    test_parser = add_experimental_parser(
        subparsers, "test", help="Run an app's on-device test"
    )
    add_app_action_args(test_parser, "test")

    configure_parser = add_experimental_parser(
        subparsers, "configure", help="Configure the target device"
    )
    configure_parser.add_argument(
        "--device",
        dest="device",
        default=None,
        metavar="DEVICE",
        help="Device name to set as the target (default: prompt to pick one)",
    )
    configure_parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show the currently configured device and exit",
    )

    argv, app_args = _split_passthrough(sys.argv[1:])
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    if args.command in ("fetch", "build", "run", "test") and (
        args.chipset or args.device
    ):
        cmd_parser = {
            "fetch": fetch_parser,
            "build": build_parser,
            "run": run_parser,
            "test": test_parser,
        }[args.command]
        flag = "--chipset" if args.chipset else "--device"
        if args.model_path is not None:
            cmd_parser.error(f"{flag} cannot be used with --model-path")
        if args.model is None and args.model_id is None:
            cmd_parser.error(f"{flag} requires --model or --model-id")
        if args.command in ("run", "test") and args.chipset:
            cmd_parser.error("--chipset cannot be used with run/test; pass --device.")

    if args.command in ("build", "run", "test") and (
        args.app is None and args.app_id is None and args.app_path is None
    ):
        cmd_parser = {"build": build_parser, "run": run_parser, "test": test_parser}[
            args.command
        ]
        cmd_parser.error("one of APP_ID_OR_PATH, --app-id or --app-path is required")

    registry_path = getattr(args, "registry", None)

    if args.command not in (
        "list",
        "info",
        "fetch",
        "build",
        "run",
        "test",
        "configure",
    ):
        parser.print_help()
        return

    try:
        if registry_path is not None and not registry_path.exists():
            raise RegistryNotFoundError(registry_path)

        registry = Registry.load(registry_path)

        if args.command == "list":
            run_list(registry)
        elif args.command == "info":
            run_info(args.app_id, registry)
        elif args.command == "fetch":
            model_asset = _resolve_model_asset(
                args.model, args.model_id, args.model_path, args.chipset, args.device
            )
            run_fetch(
                args.app_id,
                args.output_dir,
                registry,
                model_asset,
                overwrite=args.overwrite,
            )
        elif args.command == "build":
            model_asset = _resolve_model_asset(
                args.model, args.model_id, args.model_path, args.chipset, args.device
            )
            app_id, app_path = _resolve_app_target(
                args.app, args.app_id, args.app_path, args.overwrite
            )
            run_build(
                app_id,
                app_path,
                args.output_dir,
                registry,
                model_asset,
                use_docker=not args.no_docker,
                clean=args.clean,
                overwrite=args.overwrite,
            )
        elif args.command in ("run", "test"):
            model_asset = _resolve_model_asset(
                args.model, args.model_id, args.model_path, args.chipset, args.device
            )
            app_id, app_path = _resolve_app_target(
                args.app, args.app_id, args.app_path, args.overwrite
            )
            run_run(
                app_id,
                app_path,
                args.output_dir,
                registry,
                model_asset,
                use_docker=not args.no_docker,
                clean=args.clean,
                overwrite=args.overwrite,
                app_args=app_args,
                test=args.command == "test",
            )
        elif args.command == "configure":
            run_configure(args.device, show=args.show)
    except QAIHubAppsError as e:
        logger.error(str(e))  # noqa: TRY400
        sys.exit(1)
    finally:
        check_for_update()


if __name__ == "__main__":  # pragma: no cover
    main()
