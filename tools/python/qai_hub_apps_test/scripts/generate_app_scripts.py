# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import re
import stat
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from tap import Tap

from qai_hub_apps_test.configs.info_yaml import AppLanguage, AppType, QAIHAAppInfo
from qai_hub_apps_test.scripts.generate_registry import RegistryScope
from qai_hub_apps_test.utils.paths import get_all_apps

HEADER = """
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.
"""

_environment = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    # Shell uses ``${#arr[@]}``; move the comment delimiter off the default ``{#``.
    comment_start_string="{##",
    comment_end_string="##}",
)

_KIND_TEMPLATE = {
    "android": ("android/build_sh.j2", "build.sh"),
    "noop": ("noop_build_sh.j2", "build.sh"),
    "noop_windows": ("noop_build_ps1.j2", "build.ps1"),
    "windows_cpp": ("windows/build_ps1.j2", "build.ps1"),
}


def _android_package(app_dir: Path) -> str:
    """Return the Android applicationId from the app's build.gradle."""
    gradle = app_dir / "build.gradle"
    match = re.search(r"""applicationId\s*=?\s*["']([^"']+)["']""", gradle.read_text())
    if match is None:
        raise SystemExit(f"Error: no applicationId found in '{gradle}'.")
    return match.group(1)


def _launch_plan(
    info: QAIHAAppInfo, app_dir: Path
) -> tuple[str, str, dict[str, object]]:
    """Return ``(template, out_filename, context)`` for an app's launch script."""
    context: dict[str, object] = {"header": HEADER, "app_id": info.id}
    if info.app_type == AppType.UBUNTU:
        return "ubuntu/launch_sh.j2", "launch.sh", context
    if info.app_type == AppType.ANDROID:
        context["package"] = _android_package(app_dir)
        return "android/launch_sh.j2", "launch.sh", context
    return "default_launch_ps1.j2", "launch.ps1", context


def _build_plan(
    info: QAIHAAppInfo, app_dir: Path
) -> tuple[str, str, dict[str, object]]:
    """Return ``(template, out_filename, context)`` for an app's build script.

    Raises ``SystemExit`` if the app's type/language has no build script.
    """
    if info.app_type == AppType.ANDROID:
        kind = "android"
    elif info.app_type == AppType.WINDOWS and AppLanguage.CPP in info.languages:
        kind = "windows_cpp"
    elif info.app_type == AppType.WINDOWS and AppLanguage.PYTHON in info.languages:
        kind = "noop_windows"
    elif AppLanguage.PYTHON in info.languages:
        kind = "noop"
    else:
        raise SystemExit(
            f"Error: no build script for '{info.id}' "
            f"(type={info.app_type.value}, "
            f"languages={[lang.value for lang in info.languages]})."
        )

    context: dict[str, object] = {
        "header": HEADER,
        "app_id": info.id,
    }
    if kind == "windows_cpp":
        sln_files = sorted(p.name for p in app_dir.glob("*.sln"))
        if len(sln_files) != 1:
            raise SystemExit(
                f"Error: expected exactly one '.sln' in '{app_dir}' for "
                f"'{info.id}', found {len(sln_files)}: {sln_files}"
            )
        context["sln"] = sln_files[0]

    template, out_filename = _KIND_TEMPLATE[kind]
    return template, out_filename, context


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def generate_app_scripts(
    app_id: str | None = None, scope: RegistryScope = RegistryScope.PRODUCTION
) -> None:
    """Generate the build and launch scripts for each in-scope app.

    Parameters
    ----------
    app_id:
        Generate for only this app id under ``scope``. If None, all apps under provided scope is used.
    scope:
        Which apps to include (see :class:`RegistryScope`).
    """
    print(f"\n{'  Generating app scripts  ':=^60}")
    print(f"Scope:     {scope.value}")
    print(f"App:       {app_id or 'all in-scope apps'}")

    all_apps = [QAIHAAppInfo.from_app(rel) for rel in get_all_apps()]
    all_apps = [(info, d) for info, d in all_apps if scope.includes(info)]
    if app_id is not None:
        all_apps = [(info, d) for info, d in all_apps if info.id == app_id]
        if not all_apps:
            raise SystemExit(
                f"Error: no app with id '{app_id}' in scope '{scope.value}'."
            )

    generated = 0
    for info, app_dir in all_apps:
        print(f"\n{f' {info.id} ':─^60}")
        for template_name, out_filename, context in (
            _build_plan(info, app_dir),
            _launch_plan(info, app_dir),
        ):
            print(f"Template:  {template_name}")
            for key, value in context.items():
                print(f"  {key}: {value}")
            content = _environment.get_template(template_name).render(context)
            _write_executable(app_dir / out_filename, content)
            print(f"Generated {app_dir / out_filename}")
            generated += 1

    print(f"\n{'  Summary  ':=^60}")
    print(f"Generated {generated} app script(s) for scope '{scope.value}'.")


class GenerateAppScriptsParser(Tap):
    app_id: str | None = None  # Generate for a single app id under scope
    scope: RegistryScope = RegistryScope.PRODUCTION  # Which apps to include


def main() -> None:
    args = GenerateAppScriptsParser().parse_args()
    generate_app_scripts(args.app_id, args.scope)


if __name__ == "__main__":
    main()
