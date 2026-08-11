# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from qai_hub_apps_test.bundlers.shell.bundle import (
    bundle_scripts,
    collect_and_rewrite_scripts,
    find_transitive_scripts,
)
from qai_hub_apps_test.utils.versions import load_versions

pytestmark = pytest.mark.bundler_unit


def test_collect_and_rewrite_bash(tmp_path: Path, dummy_scripts_path: Path) -> None:
    script = tmp_path / "install_runtime.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"source {dummy_scripts_path}/apt_utils.sh\n"
        "install_apt_pkg libcairo2-dev\n"
    )
    rewritten, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert 'source "$(dirname "${BASH_SOURCE[0]}")/scripts/apt_utils.sh"' in rewritten
    assert dummy_scripts_path / "apt_utils.sh" in refs


def test_collect_and_rewrite_ps1_dot_source(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    script = tmp_path / "install_runtime.ps1"
    script.write_text(
        f'. "{dummy_scripts_path}/winget_utils.ps1"\nInstall-WingetPackage x\n'
    )
    rewritten, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert "winget_utils.ps1" in rewritten
    assert "$PSScriptRoot" in rewritten
    assert dummy_scripts_path / "winget_utils.ps1" in refs


def test_collect_and_rewrite_ps1_call_operator(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    script = tmp_path / "install_runtime.ps1"
    script.write_text(
        f'& "{dummy_scripts_path}/pip_utils.ps1"\nInstall-PipDeps r.txt\n'
    )
    rewritten, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert "pip_utils.ps1" in rewritten
    assert "$PSScriptRoot" in rewritten
    assert dummy_scripts_path / "pip_utils.ps1" in refs


def test_collect_and_rewrite_bash_relative_path(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    """Relative paths still resolve after copytree to a temp dir where direct resolution breaks."""
    import os
    import shutil

    # Original app dir — relative path from here is valid
    app_dir = tmp_path / "apps" / "myapp"
    app_dir.mkdir(parents=True)
    relative_source = os.path.relpath(dummy_scripts_path / "apt_utils.sh", app_dir)
    (app_dir / "install_runtime.sh").write_text(f"source {relative_source}\n")

    # Simulate copytree into a temp dir — direct resolution now breaks
    temp_copy = tmp_path / "tmp_bundle" / "myapp"
    shutil.copytree(app_dir, temp_copy)
    script = temp_copy / "install_runtime.sh"

    # Direct resolution would give tmp_bundle/myapp/../apps/_shared/... which doesn't exist
    assert (
        not (temp_copy / relative_source).resolve().exists()
        or not (temp_copy.parent / relative_source).resolve().exists()
    )

    rewritten, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert 'source "$(dirname "${BASH_SOURCE[0]}")/scripts/apt_utils.sh"' in rewritten
    assert dummy_scripts_path / "apt_utils.sh" in refs


def test_collect_and_rewrite_ps1_backslash_relative_path(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    r"""Windows backslash relative paths (..\\..\\_shared\\scripts\\foo.ps1) resolve correctly."""
    import os

    # Place app as sibling of _shared so relative path contains _shared/scripts/
    app_dir = tmp_path / "apps" / "myapp"
    app_dir.mkdir(parents=True)
    script = app_dir / "install_runtime.ps1"
    rel = os.path.relpath(dummy_scripts_path / "winget_utils.ps1", app_dir).replace(
        "/", "\\"
    )
    script.write_text(f'. "{rel}"\n')
    rewritten, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert "winget_utils.ps1" in rewritten
    assert "$PSScriptRoot" in rewritten
    assert dummy_scripts_path / "winget_utils.ps1" in refs


def test_collect_and_rewrite_var_prefix_resolves(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    """$VAR/file.sh pattern resolves tail against shared_scripts_root."""
    script = tmp_path / "install_runtime.sh"
    script.write_text('source "$_SOME_DIR/apt_utils.sh"\n')
    rewritten, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert 'source "$(dirname "${BASH_SOURCE[0]}")/scripts/apt_utils.sh"' in rewritten
    assert dummy_scripts_path / "apt_utils.sh" in refs


def test_collect_and_rewrite_pure_var_skipped_silently(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    """Pure $VARNAME (no path separator) is silently skipped — no warning."""
    script = tmp_path / "install_runtime.sh"
    original = 'source "$_VERSIONS_FILE"\necho done\n'
    script.write_text(original)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rewritten, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert rewritten == original
    assert len(refs) == 0


def test_collect_and_rewrite_warns_when_candidate_not_found(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    """Warn when a source line resolves into shared_scripts_root but file doesn't exist."""
    script = tmp_path / "install_runtime.sh"
    script.write_text(f"source {dummy_scripts_path}/nonexistent.sh\n")
    with pytest.warns(UserWarning, match="nonexistent.sh"):
        _, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert len(refs) == 0


def test_collect_and_rewrite_ignores_lines_outside_shared_scripts_root(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    other = tmp_path / "other_helper.sh"
    other.write_text("echo hi\n")
    script = tmp_path / "install_runtime.sh"
    original = f"source {other}\necho done\n"
    script.write_text(original)
    rewritten, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert rewritten == original
    assert len(refs) == 0


def test_collect_and_rewrite_multiple_refs(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    script = tmp_path / "install_runtime.sh"
    script.write_text(
        f"source {dummy_scripts_path}/apt_utils.sh\n"
        f"source {dummy_scripts_path}/load_versions.sh\n"
    )
    _, refs = collect_and_rewrite_scripts(script, dummy_scripts_path)
    assert len(refs) == 2
    assert dummy_scripts_path / "apt_utils.sh" in refs
    assert dummy_scripts_path / "load_versions.sh" in refs


def test_find_transitive_includes_direct_ref(dummy_scripts_path: Path) -> None:
    result = find_transitive_scripts(
        dummy_scripts_path / "apt_utils.sh", dummy_scripts_path
    )
    assert dummy_scripts_path / "apt_utils.sh" in result


def test_find_transitive_follows_deps(dummy_scripts_path: Path) -> None:
    # apt_utils.sh sources load_versions.sh (set up in fixture)
    result = find_transitive_scripts(
        dummy_scripts_path / "apt_utils.sh", dummy_scripts_path
    )
    assert dummy_scripts_path / "load_versions.sh" in result


def test_find_transitive_leaf_returns_only_itself(dummy_scripts_path: Path) -> None:
    result = find_transitive_scripts(
        dummy_scripts_path / "load_versions.sh", dummy_scripts_path
    )
    assert result == {dummy_scripts_path / "load_versions.sh"}


def test_bundle_scripts_copies_shared_scripts(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    (app_dir / "install_runtime.sh").write_text(
        f"source {dummy_scripts_path}/apt_utils.sh\n"
    )
    bundle_scripts(app_dir, dummy_scripts_path)
    assert (app_dir / "scripts" / "apt_utils.sh").exists()
    assert (app_dir / "scripts" / "load_versions.sh").exists()


def test_bundle_scripts_copies_versions_env(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    (app_dir / "install_runtime.sh").write_text(
        f"source {dummy_scripts_path}/apt_utils.sh\n"
    )
    bundle_scripts(app_dir, dummy_scripts_path)
    assert "PYTHON_VERSION" in (app_dir / "scripts" / "versions.env").read_text()


def test_bundle_scripts_merges_versions_override(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    (app_dir / "install_runtime.sh").write_text(
        f"source {dummy_scripts_path}/apt_utils.sh\n"
    )
    # An override present at the bundle root (as the app copytree would place it).
    (app_dir / "versions.override.env").write_text(
        'PYTHON_VERSION="3.12"\nNEW_KEY="abc"\n'
    )
    bundle_scripts(app_dir, dummy_scripts_path)
    versions_env_file = app_dir / "scripts" / "versions.env"
    merged = load_versions(versions_env_file)
    # Overridden key wins, new key added, others preserved.
    assert merged["PYTHON_VERSION"] == "3.12"
    assert merged["NEW_KEY"] == "abc"
    assert merged["TF_LITE_VERSION"] == "2.17.0"
    # No duplicate keys in the written file.
    text = (versions_env_file).read_text()
    assert text.count("PYTHON_VERSION=") == 1
    # Override is fully resolved into versions.env and removed from the bundle.
    assert not (app_dir / "versions.override.env").exists()


def test_bundle_scripts_versions_env_matches_global_without_override(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    (app_dir / "install_runtime.sh").write_text(
        f"source {dummy_scripts_path}/apt_utils.sh\n"
    )
    bundle_scripts(app_dir, dummy_scripts_path)
    # No override -> bundled versions.env has the same keys/values as the global.
    assert load_versions(app_dir / "scripts" / "versions.env") == load_versions(
        dummy_scripts_path / "versions.env"
    )


def test_bundle_scripts_rewrites_source_lines(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    install = app_dir / "install_runtime.sh"
    install.write_text(f"source {dummy_scripts_path}/apt_utils.sh\n")
    bundle_scripts(app_dir, dummy_scripts_path)
    assert (
        'source "$(dirname "${BASH_SOURCE[0]}")/scripts/apt_utils.sh"'
        in install.read_text()
    )


def test_bundle_scripts_unreferenced_not_copied(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    (app_dir / "install_runtime.sh").write_text(
        f"source {dummy_scripts_path}/apt_utils.sh\n"
    )
    bundle_scripts(app_dir, dummy_scripts_path)
    assert not (app_dir / "scripts" / "unreferenced.sh").exists()


def test_bundle_scripts_raises_when_versions_env_missing(tmp_path: Path) -> None:
    shared_scripts_root = tmp_path / "scripts"
    shared_scripts_root.mkdir()
    (shared_scripts_root / "apt_utils.sh").write_text("# apt\n")
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    (app_dir / "install_runtime.sh").write_text(
        f"source {shared_scripts_root}/apt_utils.sh\n"
    )
    with pytest.raises(FileNotFoundError, match=re.escape("versions.env")):
        bundle_scripts(app_dir, shared_scripts_root)


def test_bundle_scripts_noop_when_no_install_scripts(
    tmp_path: Path, dummy_scripts_path: Path
) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    bundle_scripts(app_dir, dummy_scripts_path)
    assert not (app_dir / "scripts").exists()
