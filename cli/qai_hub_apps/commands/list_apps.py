# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace

from qai_hub_models_cli.utils import build_table

from qai_hub_apps.registry import App, AppFilter, Registry

# One cell renderer per column, labelled as in App.detail_fields().
_CELLS: dict[str, Callable[[App], str]] = {
    "ID": lambda app: (app.id or app.name) or "",
    "Name": lambda app: app.name,
    "Domain": lambda app: app.domain or "",
    "Languages": lambda app: ", ".join(lang.value for lang in app.languages),
    "Type": lambda app: app.app_type.value,
    "Runtime": lambda app: ", ".join(app.runtime),
    "Use Case": lambda app: app.use_case or "",
    "Precision": lambda app: ", ".join(app.precisions),
    "Models": lambda app: ", ".join(app.related_models),
    "Supported Devices": lambda app: ", ".join(app.supported_devices),
}

_BASE_COLUMNS = ("ID", "Name", "Domain", "Languages")

# Filtering on a dimension that is not already a column adds it, so the values
# that were matched on are visible. Order here is the order in the table.
_FILTER_COLUMNS: tuple[tuple[str, Callable[[AppFilter], bool]], ...] = (
    ("Type", lambda f: bool(f.app_types)),
    ("Runtime", lambda f: bool(f.runtimes)),
    ("Use Case", lambda f: bool(f.use_cases)),
    ("Precision", lambda f: bool(f.precisions)),
    ("Models", lambda f: bool(f.models)),
    ("Supported Devices", lambda f: f.device is not None),
)

# build_table wraps exactly one column; give it the widest list-valued one.
_WRAP_PRECEDENCE = ("Models", "Supported Devices", "Runtime")


def _columns_for(app_filter: AppFilter) -> list[str]:
    """Return the table's columns for *app_filter*."""
    return [
        *_BASE_COLUMNS,
        *(name for name, is_active in _FILTER_COLUMNS if is_active(app_filter)),
    ]


def _wrap_target(columns: list[str]) -> tuple[str, bool]:
    """Return the ``(wrap_column, wrap_on_commas)`` to render *columns* with."""
    for name in _WRAP_PRECEDENCE:
        if name in columns:
            return name, True
    return "Name", False


def _vocabulary_hints(app_filter: AppFilter, apps: Iterable[App]) -> list[str]:
    """Return an ``Available <dimension>: ...`` line per registry-defined filter used."""
    apps = list(apps)
    hints: dict[str, set[str]] = {}
    if app_filter.runtimes:
        others = replace(app_filter, runtimes=frozenset())
        hints["runtimes"] = {
            r for app in apps if others.matches(app) for r in app.runtime
        }
    if app_filter.domains:
        others = replace(app_filter, domains=frozenset())
        hints["domains"] = {
            app.domain for app in apps if others.matches(app) if app.domain
        }
    if app_filter.use_cases:
        others = replace(app_filter, use_cases=frozenset())
        hints["use cases"] = {
            app.use_case for app in apps if others.matches(app) if app.use_case
        }
    if app_filter.precisions:
        others = replace(app_filter, precisions=frozenset())
        hints["precisions"] = {
            p for app in apps if others.matches(app) for p in app.precisions
        }
    return [
        f"Available {label}: {', '.join(sorted(values))}."
        for label, values in hints.items()
        if values
    ]


def run_list(registry: Registry, app_filter: AppFilter | None = None) -> None:
    """List the registry's apps, narrowed by *app_filter* when one is given."""
    app_filter = app_filter or AppFilter()
    apps = registry.filter(app_filter)

    if not apps and not app_filter.is_empty():
        print("No apps match those filters.")
        for line in _vocabulary_hints(app_filter, registry.apps):
            print(line)
        print("See all apps:\n  qai-hub-apps list")
        return

    columns = _columns_for(app_filter)
    wrap_column, wrap_on_commas = _wrap_target(columns)
    print(
        build_table(
            columns,
            [[_CELLS[column](app) for column in columns] for app in apps],
            wrap_column=wrap_column,
            wrap_on_commas=wrap_on_commas,
            title="Qualcomm\u00ae AI Hub Apps",
        )
    )
    if app_filter.is_empty():
        print(f"Total: {len(apps)} apps")
    else:
        print(f"Total: {len(apps)} of {len(registry.apps)} apps")


def run_info(app_id: str, registry: Registry) -> None:
    app = registry.find_by_id(app_id)
    print(app)
