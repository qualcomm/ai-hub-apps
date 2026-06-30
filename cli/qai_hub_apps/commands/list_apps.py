# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models_cli.utils import build_table

from qai_hub_apps.registry import Registry


def run_list(registry: Registry) -> None:
    apps = list(registry.apps)
    rows = [
        [
            (app.id or app.name) or "",
            app.name,
            app.domain or "",
            ", ".join(lang.value for lang in app.languages),
        ]
        for app in apps
    ]
    print(
        build_table(
            ["ID", "Name", "Domain", "Languages"],
            rows,
            wrap_column="Name",
            title="Qualcomm\u00ae AI Hub Apps",
        )
    )
    print(f"Total: {len(apps)} apps")


def run_info(app_id: str, registry: Registry) -> None:
    app = registry.find_by_id(app_id)
    print(app)
