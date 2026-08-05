# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Bundle shared shell scripts into a standalone app directory.

The bundler:
  1. For each install_*.sh / install_*.ps1 / test.{sh,ps1} / build.{sh,ps1}:
     rewrites source lines to the bundle-local scripts/ prefix and collects
     direct shared refs.
  2. Transitively copies all referenced shared scripts to out_dir/scripts/.
  3. Copies apps/_shared/scripts/versions.env to out_dir/scripts/versions.env.
"""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path

from qai_hub_apps_test.bundlers.shell.script_resolver import resolve_scripts_root

# Match: source <path>  (bash)
_BASH_SOURCE_RE = re.compile(r"""^\s*source\s+["']?(\S+?)["']?\s*$""")

# Match: . <path>  or  & <path>  (PowerShell dot-source / call operator)
_PS_SOURCE_RE = re.compile(r"""^\s*[.&]\s+["']?(\S+?)["']?\s*$""")


def collect_and_rewrite_scripts(
    script_path: Path, shared_scripts_root: Path
) -> tuple[str, list[Path]]:
    r"""Read a script, rewrite source lines, and return the new content + direct refs.

    For each line that sources a file under ``shared_scripts_root``, the path is
    rewritten to the bundle-local ``scripts/`` prefix and the referenced file
    is recorded. Subdirectory structure within ``shared_scripts_root`` is preserved.

    Bash (``.sh``):
        ``source <old_path>`` ->
        ``source "$(dirname "${BASH_SOURCE[0]}")/scripts/<rel>"``

    PowerShell (``.ps1``):
        ``. <old_path>`` or ``& <old_path>`` ->
        ``. "$PSScriptRoot\\scripts\\<rel>"``

    Parameters
    ----------
    script_path:
        Path of the script to read and process.
    shared_scripts_root:
        Shared scripts directory used to identify which source lines to rewrite.

    Returns
    -------
    tuple[str, list[Path]]
        ``(rewritten_content, direct_refs)`` where ``direct_refs`` is the list
        of shared script paths directly referenced by this script.
    """
    is_bash = script_path.suffix == ".sh"
    pattern = _BASH_SOURCE_RE if is_bash else _PS_SOURCE_RE
    shared_scripts_root_resolved = shared_scripts_root.resolve()

    direct_refs: list[Path] = []
    new_lines: list[str] = []

    # Fallback marker: install scripts reference shared scripts as relative paths
    # (e.g. ../../_shared/scripts/foo.sh). After copytree into a temp dir the normal
    # resolution breaks, so we extract the tail after this marker and resolve from
    # shared_scripts_root instead.
    _SHARED_SCRIPTS_MARKER = "_shared/scripts/"

    for lineno, line in enumerate(
        script_path.read_text(encoding="utf-8").splitlines(keepends=True), start=1
    ):
        m = pattern.match(line.strip())
        if m:
            raw_path = m.group(1)
            normalized_path = raw_path.replace("\\", "/")
            candidates: list[Path] = []
            if not raw_path.startswith("$"):
                candidates.append((script_path.parent / raw_path).resolve())
                if (
                    not Path(raw_path).is_absolute()
                    and _SHARED_SCRIPTS_MARKER in normalized_path
                ):
                    tail = normalized_path.split(_SHARED_SCRIPTS_MARKER, 1)[1]
                    candidates.append((shared_scripts_root / tail).resolve())
            else:
                # $VAR/relative/path.sh or $VAR\relative\path.ps1 — strip the
                # variable prefix up to the first path separator and resolve the
                # tail from shared_scripts_root. If no separator exists the path is
                # a pure variable reference ($VARNAME) which cannot be statically
                # resolved; skip silently.
                sep_idx = min(
                    (raw_path.find(c) for c in "/\\" if c in raw_path),
                    default=-1,
                )
                if sep_idx != -1:
                    tail = raw_path[sep_idx + 1 :]
                    if tail:
                        candidates.append((shared_scripts_root / tail).resolve())

            for candidate in candidates:
                if (
                    candidate.is_relative_to(shared_scripts_root_resolved)
                    and candidate.exists()
                ):
                    rel = candidate.relative_to(shared_scripts_root_resolved)
                    direct_refs.append(candidate)
                    eol = "\n" if line.endswith("\n") else ""
                    if is_bash:
                        line = f'source "$(dirname "${{BASH_SOURCE[0]}}")/scripts/{rel.as_posix()}"{eol}'
                    else:
                        line = f'. "$PSScriptRoot\\scripts\\{rel!s}"{eol}'
                    break
            else:
                if candidates:
                    warnings.warn(
                        f"Source line '{raw_path}' in '{script_path}:{lineno}' did not resolve "
                        f"to a file under '{shared_scripts_root}'. It will be skipped.",
                        stacklevel=2,
                    )
        new_lines.append(line)

    return "".join(new_lines), direct_refs


def find_transitive_scripts(direct_ref: Path, shared_scripts_root: Path) -> set[Path]:
    """Return all shared scripts transitively reachable from a single direct ref.

    Performs a DFS from ``direct_ref``, following source / dot-source lines
    within ``shared_scripts_root``. The returned set includes ``direct_ref`` itself.

    Parameters
    ----------
    direct_ref:
        The directly-referenced shared script to start from.
    shared_scripts_root:
        Shared scripts directory — only refs resolving here are followed.

    Returns
    -------
    set[Path]
        All shared script paths reachable from ``direct_ref`` (inclusive).
    """
    visited: set[Path] = set()
    stack = [direct_ref]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        _, deps = collect_and_rewrite_scripts(current, shared_scripts_root)
        stack.extend(dep for dep in deps if dep not in visited)

    return visited


def bundle_scripts(out_dir: Path, shared_scripts_root: Path | None = None) -> None:
    """Process install/test scripts, copy shared scripts and versions.env to scripts/.

    Operates on ``out_dir`` after the app source has already been copied there.

    Parameters
    ----------
    out_dir:
        Bundle staging directory containing app source + install scripts.
    shared_scripts_root:
        The shared scripts directory (``apps/_shared/scripts/``).
        Auto-resolved from the repository structure if None.
    """
    shared_scripts_root = resolve_scripts_root(shared_scripts_root)
    app_scripts = (
        list(out_dir.glob("install_*.sh"))
        + list(out_dir.glob("install_*.ps1"))
        + list(out_dir.glob("test.sh"))
        + list(out_dir.glob("test.ps1"))
        + list(out_dir.glob("build.sh"))
        + list(out_dir.glob("build.ps1"))
    )
    if not app_scripts:
        print(
            "No install_*.sh / install_*.ps1 / test.sh / test.ps1 / build.sh / "
            "build.ps1 found; skipping shell script bundling."
        )
        return

    print(f"Found {len(app_scripts)} script(s) to process.")

    scripts_dir = out_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Rewrite each script and collect direct shared refs
    all_direct_refs: set[Path] = set()
    for script in app_scripts:
        rewritten, direct_refs = collect_and_rewrite_scripts(
            script, shared_scripts_root
        )
        script.write_text(rewritten, encoding="utf-8")
        all_direct_refs.update(direct_refs)

    if not all_direct_refs:
        print("No shared script references found; install scripts need no rewriting.")
        return

    # Collect + copy all transitive shared scripts
    all_shared: set[Path] = set()
    for ref in all_direct_refs:
        all_shared.update(find_transitive_scripts(ref, shared_scripts_root))

    for shared in sorted(all_shared):
        rel = shared.relative_to(shared_scripts_root)
        dest = scripts_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shared, dest)

    print(f"Found {len(all_shared)} shared script(s) to include.")

    versions_env_src = shared_scripts_root / "versions.env"
    if not versions_env_src.exists():
        raise FileNotFoundError(
            f"versions.env not found at '{versions_env_src}'. "
            "It is required when shared scripts are referenced."
        )
    shutil.copy2(versions_env_src, scripts_dir / "versions.env")
