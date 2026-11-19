# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Regenerate every committed generated file, as CI checks them."""

from __future__ import annotations

from tap import Tap

from qai_hub_apps._internal.configs.asset_bases_yaml import AssetBases
from qai_hub_apps._internal.configs.info_yaml import AppInfoInternal
from qai_hub_apps._internal.scripts.generate_app_scripts import generate_app_scripts
from qai_hub_apps._internal.scripts.generate_registry import (
    MIN_CLI_VERSION,
    RegistryScope,
    generate_registry,
    read_cli_version,
)
from qai_hub_apps._internal.scripts.generate_shared_readme import (
    generate_shared_readme,
)
from qai_hub_apps._internal.utils.paths import REPOSITORY_ROOT, get_all_apps


class RunCodegenParser(Tap):
    pass


def main() -> None:
    RunCodegenParser().parse_args()
    generate_registry(
        output_dir=REPOSITORY_ROOT / "cli" / "qai_hub_apps",
        all_apps=[AppInfoInternal.from_app(rel) for rel in get_all_apps()],
        repo_base=AssetBases.load().app_repo_base,
        ref="main",
        cli_version=read_cli_version(),
        min_cli_version=MIN_CLI_VERSION,
    )
    generate_app_scripts(scope=RegistryScope.TEST)
    generate_shared_readme()


if __name__ == "__main__":
    main()
