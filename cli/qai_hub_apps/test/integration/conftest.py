# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from qai_hub_apps.main import main
from qai_hub_apps.registry.base import Registry

_SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Overwrite snapshot files with current output instead of comparing.",
    )


@pytest.fixture
def snapshot(request: pytest.FixtureRequest) -> Callable[[str, str], None]:
    update = request.config.getoption("--update-snapshots")

    def _compare_or_update(name: str, actual: str) -> None:
        path = _SNAPSHOTS_DIR / name
        if update:
            _SNAPSHOTS_DIR.mkdir(exist_ok=True)
            path.write_text(actual, encoding="utf-8")
        else:
            if not path.exists():
                raise FileNotFoundError(
                    f"Snapshot file not found: {path}\n"
                    f"Run `pytest qai_hub_apps/test/integration/ -m integration --no-cov --update-snapshots` to create it."
                )
            expected = path.read_text(encoding="utf-8")
            assert actual == expected, (
                f"Snapshot mismatch for '{name}'.\n"
                f"Run `pytest qai_hub_apps/test/integration/ -m integration --no-cov --update-snapshots` to accept the new output."
            )

    return _compare_or_update


_TWO_APP_REGISTRY = """\
schema_version: '1.1'
min_cli_version: 0.0.1
apps:
- name: Whisper Windows
  id: whisper_windows_py
  status: published
  headline: Speech to text on Windows
  description: Run Whisper on-device using ONNX.
  domain: Audio
  use_case: Speech Recognition
  app_repo_url: https://github.com/qualcomm/ai-hub-apps/tree/main/apps/whisper_windows_py
  app_type: windows
  runtime: onnx
  related_models: [whisper_base]
  precisions: [float]
  model_file_dir: models
  url:
    source: https://example.com/whisper_windows_py.zip
- name: Stable Diffusion
  id: stable_diffusion_py
  status: published
  headline: Image generation on Windows
  description: Run Stable Diffusion on-device using ONNX.
  domain: Generative AI
  use_case: Image Generation
  app_repo_url: https://github.com/qualcomm/ai-hub-apps/tree/main/apps/stable_diffusion_py
  app_type: windows
  runtime: onnx
  related_models: [stable_diffusion]
  precisions: [float]
"""


@pytest.fixture(autouse=True)
def reset_registry_singleton():
    Registry._instance = None
    yield
    Registry._instance = None


@pytest.fixture
def whisper_app_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "whisper_windows_py.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("requirements.txt", "onnxruntime\n")
        zf.writestr("run.py", "# Whisper app\n")
    return zip_path


@pytest.fixture
def exported_model_dir(tmp_path: Path) -> Path:
    """A locally-exported model directory (metadata.json + model file)."""
    import json

    export = tmp_path / "exported_model"
    export.mkdir()
    (export / "whisper_base.onnx").write_text("MODEL\n")
    (export / "metadata.json").write_text(
        json.dumps(
            {"model_id": "whisper_base", "model_files": {"whisper_base.onnx": {}}}
        )
    )
    return export


@pytest.fixture
def two_app_registry(tmp_path: Path) -> Path:
    p = tmp_path / "registry.yaml"
    p.write_text(_TWO_APP_REGISTRY)
    return p


def run_cli(argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["qai-hub-apps", *argv])
    main()
