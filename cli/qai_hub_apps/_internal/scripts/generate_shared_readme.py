# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import ast
import re
from pathlib import Path

from tap import Tap

from qai_hub_apps._internal.utils.paths import SHARED_UTILS_ROOT

HEADER = """\
<!-- THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY. -->
<!-- Regenerate with: python -m qai_hub_apps._internal.scripts.run_codegen -->
# Shared App Utilities (`apps/_shared/`)

Code shared across apps. Each description is the file's `Description:` header
line, or the first line of the module docstring for Python.
"""

_DESCRIPTION_RE = re.compile(r"^[^\w\n]*Description:[ \t]*(\S.*)$", re.MULTILINE)


def _describe(path: Path) -> str:
    """Return the first docstring line for Python, else the ``Description:`` line."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        docstring = ast.get_docstring(ast.parse(text))
        description = docstring.splitlines()[0] if docstring else None
    else:
        match = _DESCRIPTION_RE.search(text)
        description = match.group(1).strip() if match else None
    if not description:
        raise SystemExit(
            f"Error: '{path}' has no description. Add a module docstring (Python) "
            "or a 'Description:' header comment line."
        )
    return description


def _link(path: Path, label: str) -> str:
    return f"[`{label}`]({path.relative_to(SHARED_UTILS_ROOT).as_posix()})"


def _rows(paths: list[Path], base: Path) -> list[str]:
    return [
        f"| {_link(p, p.relative_to(base).as_posix())} | {_describe(p)} |"
        for p in sorted(paths)
    ]


def _scripts_section() -> list[str]:
    scripts_dir = SHARED_UTILS_ROOT / "scripts"
    paths = [
        p for p in scripts_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    ]
    return [
        "## Scripts (`scripts/`)",
        "",
        "Sourced by app install/test scripts and the generated build/launch scripts, "
        "never executed directly. Bundling copies the ones an app uses into its "
        "`scripts/` directory.",
        "",
        "| File | Description |",
        "|------|-------------|",
        *_rows(paths, scripts_dir),
    ]


def _python_section() -> list[str]:
    python_dir = SHARED_UTILS_ROOT / "python"
    package_dir = python_dir / "qai_hub_apps_utils"
    rows = []
    for path in sorted([*python_dir.glob("*.py"), *package_dir.glob("*.py")]):
        if path.name == "__init__.py":
            continue
        requirements = package_dir / "requirements" / f"requirements-{path.stem}.txt"
        extra = f"`[{path.stem}]`" if requirements.is_file() else ""
        label = path.relative_to(python_dir).as_posix()
        rows.append(f"| {_link(path, label)} | {extra} | {_describe(path)} |")
    return [
        "## Python (`python/`)",
        "",
        "`qai_hub_apps_utils/` is installed as `qai-hub-apps-utils`; each module's "
        "third-party dependencies are the pip extra listed below.",
        "",
        "| File | Extra | Description |",
        "|------|-------|-------------|",
        *rows,
    ]


def _other_section() -> list[str]:
    paths = [
        p
        for subdir in SHARED_UTILS_ROOT.iterdir()
        if subdir.is_dir() and subdir.name not in ("python", "scripts")
        for p in subdir.rglob("*")
        if p.is_file()
    ]
    return [
        "## Other shared files",
        "",
        "| File | Description |",
        "|------|-------------|",
        *_rows(paths, SHARED_UTILS_ROOT),
    ]


def generate_shared_readme(output: Path = SHARED_UTILS_ROOT / "README.md") -> None:
    """Write the README listing every shared utility file and its description.

    Parameters
    ----------
    output:
        Path of the README to write.
    """
    sections = [_scripts_section(), _python_section(), _other_section()]
    content = "\n\n".join([HEADER.rstrip("\n"), *("\n".join(s) for s in sections)])
    output.write_text(content + "\n", encoding="utf-8")
    print(f"Generated {output}")


class GenerateSharedReadmeParser(Tap):
    output: Path = SHARED_UTILS_ROOT / "README.md"  # Path of the README to write


def main() -> None:
    args = GenerateSharedReadmeParser().parse_args()
    generate_shared_readme(args.output)


if __name__ == "__main__":
    main()
