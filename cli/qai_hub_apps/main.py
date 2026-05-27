# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import sys
from pathlib import Path

from qai_hub_apps import __version__, _is_dev
from qai_hub_apps.commands.fetch import run_fetch
from qai_hub_apps.commands.list_apps import run_info, run_list
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.errors import QAIHubAppsError, RegistryNotFoundError
from qai_hub_apps.registry import Registry


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
    fetch_parser.add_argument(
        "--model",
        dest="model",
        default=None,
        metavar="MODEL_ID",
        help="Also download the specified model (must be supported by the app)",
    )
    fetch_parser.add_argument(
        "--chipset",
        dest="chipset",
        default=None,
        metavar="CHIPSET",
        help="Chipset to target when downloading model (must be supported by the app)",
    )

    args = parser.parse_args()

    if args.command == "fetch" and args.chipset and not args.model:
        fetch_parser.error("--chipset requires --model")

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
            model_asset = (
                ModelAsset(model_id=args.model, chipset=args.chipset)
                if args.model is not None
                else None
            )
            run_fetch(args.app_id, args.output_dir, registry, model_asset)
    except QAIHubAppsError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
