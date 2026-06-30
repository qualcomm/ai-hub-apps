# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import sys
from pathlib import Path

from qai_hub_apps import __version__, _is_dev
from qai_hub_apps.commands.fetch import run_fetch
from qai_hub_apps.commands.list_apps import run_info, run_list
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.errors import QAIHubAppsError, RegistryNotFoundError
from qai_hub_apps.logging_utils import configure_logging
from qai_hub_apps.registry import Registry

logger = logging.getLogger(__name__)


def _resolve_model_asset(
    model: str | None,
    model_id: str | None,
    model_path: Path | None,
    chipset: str | None,
) -> ModelAsset | None:
    """Build a ModelAsset from the --model / --model-id / --model-path args.

    --model-id and --model-path are explicit. Plain --model is auto-resolved: an
    existing path is treated as a local export, otherwise as a model ID. (A
    --model value that is both a supported model and a path is rejected later,
    in App.fetch, with a hint to use the explicit flags.)
    """
    if model_id is not None:
        return ModelAsset(model_id=model_id, chipset=chipset)
    if model_path is not None:
        # Resolve to an absolute path so it can never collide with a model id.
        return ModelAsset(path=model_path.resolve())
    if model is None:
        return None
    path = Path(model)
    if path.exists():
        if chipset is not None:
            logger.warning("--chipset is ignored when --model is a local path.")
        return ModelAsset(path=path)
    return ModelAsset(model_id=model, chipset=chipset)


def main() -> None:
    epilog = (
        "Examples:\n"
        "  qai-hub-apps list                   List all available apps\n"
        "  qai-hub-apps info <app_id>          Show details for an app\n"
        "  qai-hub-apps fetch <app_id>         Download an app's source\n"
    )
    parser = argparse.ArgumentParser(
        prog="qai-hub-apps",
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
    fetch_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory (default: current directory)",
    )
    model_group = fetch_parser.add_mutually_exclusive_group()
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
    fetch_parser.add_argument(
        "--chipset",
        dest="chipset",
        default=None,
        metavar="CHIPSET",
        help="Chipset to target when downloading model (must be supported by the app)",
    )

    args = parser.parse_args()

    configure_logging(args.log_level)

    if args.command == "fetch" and args.chipset:
        if args.model_path is not None:
            fetch_parser.error("--chipset cannot be used with --model-path")
        if args.model is None and args.model_id is None:
            fetch_parser.error("--chipset requires --model or --model-id")

    registry_path = getattr(args, "registry", None)

    if args.command not in ("list", "info", "fetch"):
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
                args.model, args.model_id, args.model_path, args.chipset
            )
            run_fetch(args.app_id, args.output_dir, registry, model_asset)
    except QAIHubAppsError as e:
        logger.error(str(e))  # noqa: TRY400
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
