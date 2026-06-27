# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from .conftest import run_cli

pytestmark = pytest.mark.integration


def test_list_output(monkeypatch, two_app_registry, capsys, snapshot):
    run_cli(["list", "--registry", str(two_app_registry)], monkeypatch)
    snapshot("list.txt", capsys.readouterr().out)


def test_info_output(monkeypatch, two_app_registry, capsys, snapshot):
    run_cli(
        ["info", "whisper_windows_py", "--registry", str(two_app_registry)], monkeypatch
    )
    snapshot("info_whisper_windows_py.txt", capsys.readouterr().out)


def test_fetch_output(
    monkeypatch, two_app_registry, tmp_path, capsys, snapshot, whisper_app_zip
):
    def fake_download(url, dest, **kwargs):
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(whisper_app_zip) as zf:
            zf.extractall(dest)
        return dest

    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.is_app_supported", lambda _: True)

    # only mock download from qai_hub_models_cli
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    run_cli(
        [
            "fetch",
            "whisper_windows_py",
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(two_app_registry),
        ],
        monkeypatch,
    )

    out = capsys.readouterr().out.replace(tmp_path.as_posix(), "<dest>")
    snapshot("fetch.txt", out)

    extracted = tmp_path / "whisper_windows_py"
    tree = "\n".join(
        str(p.relative_to(extracted)) for p in sorted(extracted.rglob("*"))
    )
    snapshot("fetch_tree.txt", tree)


def test_fetch_with_local_model_dir(
    monkeypatch,
    two_app_registry,
    tmp_path,
    capsys,
    snapshot,
    whisper_app_zip,
    exported_model_dir,
):
    def fake_download(url, dest, **kwargs):
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(whisper_app_zip) as zf:
            zf.extractall(dest)
        return dest

    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.is_app_supported", lambda _: True)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)
    # A local --model must never trigger an asset download.
    monkeypatch.setattr(
        "qai_hub_apps.registry.base.get_asset_url",
        lambda **kwargs: pytest.fail(
            "get_asset_url should not be called for a local model"
        ),
    )

    run_cli(
        [
            "fetch",
            "whisper_windows_py",
            "--model",
            str(exported_model_dir),
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(two_app_registry),
        ],
        monkeypatch,
    )

    out = capsys.readouterr().out.replace(tmp_path.as_posix(), "<dest>")
    snapshot("fetch_local_model.txt", out)

    extracted = tmp_path / "whisper_windows_py"
    tree = "\n".join(
        str(p.relative_to(extracted)) for p in sorted(extracted.rglob("*"))
    )
    snapshot("fetch_local_model_tree.txt", tree)


def test_fetch_ambiguous_model_exits(
    monkeypatch, two_app_registry, tmp_path, capsys, whisper_app_zip
):
    def fake_download(url, dest, **kwargs):
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(whisper_app_zip) as zf:
            zf.extractall(dest)
        return dest

    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: False)
    monkeypatch.setattr("qai_hub_apps.registry.base.is_app_supported", lambda _: True)
    monkeypatch.setattr("qai_hub_apps.registry.base.download", fake_download)

    # A directory in the cwd named exactly like the supported model 'whisper_base'.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "whisper_base").mkdir()

    with pytest.raises(SystemExit):
        run_cli(
            [
                "fetch",
                "whisper_windows_py",
                "--model",
                "whisper_base",
                "--output-dir",
                str(tmp_path),
                "--registry",
                str(two_app_registry),
            ],
            monkeypatch,
        )

    assert "--model-id or --model-path" in capsys.readouterr().out


def test_fetch_dev_output(monkeypatch, two_app_registry, tmp_path, capsys, snapshot):
    def fake_bundle_app(app_id, dest, make_zip=False):
        out_dir = Path(dest) / app_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "requirements.txt").write_text("torch\n")
        (out_dir / "run.py").write_text("# Stable Diffusion app\n")

    monkeypatch.setattr("qai_hub_apps.registry.base._is_dev", lambda: True)
    monkeypatch.setattr("qai_hub_apps.registry.base.is_app_supported", lambda _: True)

    # only mock bundler from qai_hub_apps_test
    monkeypatch.setattr("qai_hub_apps.registry.base._bundle_app", fake_bundle_app)

    run_cli(
        [
            "fetch",
            "stable_diffusion_py",
            "--output-dir",
            str(tmp_path),
            "--registry",
            str(two_app_registry),
        ],
        monkeypatch,
    )

    out = capsys.readouterr().out.replace(tmp_path.as_posix(), "<dest>")
    snapshot("fetch_dev.txt", out)

    extracted = tmp_path / "stable_diffusion_py"
    tree = "\n".join(
        str(p.relative_to(extracted)) for p in sorted(extracted.rglob("*"))
    )
    snapshot("fetch_dev_tree.txt", tree)
