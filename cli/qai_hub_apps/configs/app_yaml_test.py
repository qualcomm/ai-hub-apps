# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from qai_hub_apps.configs.app_yaml import (
    AppInfo,
    AppLanguage,
    AppType,
    AppUrl,
)

_REQUIRED_FIELDS: dict = dict(
    name="My App",
    id="my_app",
    status="published",
    headline="Short headline",
    description="Longer description",
    domain="Audio",
    use_case="Speech Recognition",
    app_repo_url="https://github.com/test/app",
    app_type="ubuntu",
    runtime="onnx",
    related_models=["whisper_base"],
    precisions=["float"],
)


def test_app_info_required_fields():
    info = AppInfo(**_REQUIRED_FIELDS)
    assert info.name == "My App"
    assert info.id == "my_app"
    assert info.app_type == AppType.UBUNTU
    assert info.runtime == ["onnx"]
    assert info.related_models == ["whisper_base"]
    assert info.precisions == ["float"]


def test_app_info_optional_url_defaults_none():
    info = AppInfo(**_REQUIRED_FIELDS)
    assert info.url is None


def test_app_info_url_when_provided():
    info = AppInfo(**_REQUIRED_FIELDS, url=AppUrl(source="https://example.com/app.zip"))
    assert info.url is not None
    assert info.url.source == "https://example.com/app.zip"


def test_app_info_optional_languages_defaults_empty():
    info = AppInfo(**_REQUIRED_FIELDS)
    assert info.languages == []


def test_app_info_optional_model_file_paths_defaults_empty():
    info = AppInfo(**_REQUIRED_FIELDS)
    assert info.model_file_paths == []


def test_app_info_optional_environment_defaults_none():
    info = AppInfo(**_REQUIRED_FIELDS)
    assert info.environment is None


def test_app_url_source_field():
    url = AppUrl(source="https://example.com/source.zip")
    assert url.source == "https://example.com/source.zip"


def test_app_type_enum_values():
    assert AppType("android") == AppType.ANDROID
    assert AppType("windows") == AppType.WINDOWS
    assert AppType("ubuntu") == AppType.UBUNTU


def test_app_language_enum_values():
    assert AppLanguage("Python") == AppLanguage.PYTHON
    assert AppLanguage("C++") == AppLanguage.CPP
    assert AppLanguage("Java") == AppLanguage.JAVA
    assert AppLanguage("Kotlin") == AppLanguage.KOTLIN


def test_extra_fields_ignored():
    info = AppInfo.model_validate(
        {**_REQUIRED_FIELDS, "unknown_field": "should_be_ignored"}
    )
    assert not hasattr(info, "unknown_field")


def test_from_yaml_loads_valid_file(tmp_path: Path):
    content = """\
name: My App
id: my_app
status: published
headline: Short headline
description: Longer description
domain: Audio
use_case: Speech Recognition
app_repo_url: https://github.com/test/app
app_type: ubuntu
runtime: onnx
related_models: [whisper_base]
precisions: [float]
languages: [Python]
"""
    p = tmp_path / "app.yaml"
    p.write_text(content)
    info = AppInfo.from_yaml(p)
    assert info.id == "my_app"
    assert AppLanguage.PYTHON in info.languages


def test_from_yaml_missing_required_field_raises(tmp_path: Path):
    # `runtime` is required — omitting it should raise a ValidationError
    content = """\
name: My App
id: my_app
status: published
headline: Short headline
description: Longer description
domain: Audio
use_case: Speech Recognition
app_repo_url: https://github.com/test/app
app_type: ubuntu
related_models: [whisper_base]
precisions: [float]
"""
    p = tmp_path / "app.yaml"
    p.write_text(content)
    with pytest.raises(ValidationError):
        AppInfo.from_yaml(p)


def test_from_yaml_invalid_enum_value_raises(tmp_path: Path):
    # `app_type` must be one of android/windows/ubuntu
    content = """\
name: My App
id: my_app
status: published
headline: Short headline
description: Longer description
domain: Audio
use_case: Speech Recognition
app_repo_url: https://github.com/test/app
app_type: not_a_real_platform
runtime: [onnx]
related_models: [whisper_base]
precisions: [float]
"""
    p = tmp_path / "app.yaml"
    p.write_text(content)
    with pytest.raises(ValidationError):
        AppInfo.from_yaml(p)
