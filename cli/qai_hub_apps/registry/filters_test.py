# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest

from qai_hub_apps.configs.app_yaml import AppLanguage, AppType
from qai_hub_apps.conftest import make_app_info
from qai_hub_apps.errors import InvalidArgumentError
from qai_hub_apps.registry.base import App
from qai_hub_apps.registry.filters import AppFilter, build_app_filter


def _app(**overrides) -> App:
    return App(make_app_info(**overrides))


def test_empty_filter_is_empty():
    assert AppFilter().is_empty()


def test_empty_filter_matches_any_app():
    assert AppFilter().matches(_app())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"app_types": frozenset({AppType.UBUNTU})},
        {"languages": frozenset({AppLanguage.PYTHON})},
        {"runtimes": frozenset({"onnx"})},
        {"domains": frozenset({"test"})},
        {"use_cases": frozenset({"testing"})},
        {"precisions": frozenset({"float"})},
        {"models": ("test",)},
        {"device": "Snapdragon 8 Elite QRD"},
    ],
)
def test_is_empty_false_for_each_dimension(kwargs):
    assert not AppFilter(**kwargs).is_empty()


def test_app_type_matches():
    assert AppFilter(app_types=frozenset({AppType.UBUNTU})).matches(_app())


def test_app_type_no_match():
    assert not AppFilter(app_types=frozenset({AppType.ANDROID})).matches(_app())


def test_app_type_matches_any_of():
    app_filter = AppFilter(app_types=frozenset({AppType.ANDROID, AppType.UBUNTU}))
    assert app_filter.matches(_app())


def test_language_matches_any_of():
    app_filter = AppFilter(languages=frozenset({AppLanguage.PYTHON, AppLanguage.GO}))
    assert app_filter.matches(_app(languages=[AppLanguage.GO]))


def test_language_no_match():
    app_filter = AppFilter(languages=frozenset({AppLanguage.KOTLIN}))
    assert not app_filter.matches(_app(languages=[AppLanguage.PYTHON]))


def test_runtime_matches_normalized_value():
    app_filter = AppFilter(runtimes=frozenset({"geniex qairt"}))
    assert app_filter.matches(_app(runtime=["geniex_qairt"]))


def test_runtime_matches_any_of():
    app_filter = AppFilter(runtimes=frozenset({"tflite", "onnx"}))
    assert app_filter.matches(_app(runtime=["onnx"]))


def test_runtime_no_match():
    assert not AppFilter(runtimes=frozenset({"tflite"})).matches(_app(runtime=["onnx"]))


def test_domain_matches_case_insensitively():
    app_filter = AppFilter(domains=frozenset({"generative ai"}))
    assert app_filter.matches(_app(domain="Generative AI"))


def test_domain_no_match():
    assert not AppFilter(domains=frozenset({"audio"})).matches(_app(domain="Test"))


def test_use_case_matches():
    assert AppFilter(use_cases=frozenset({"testing"})).matches(_app())


def test_use_case_no_match():
    app_filter = AppFilter(use_cases=frozenset({"object detection"}))
    assert not app_filter.matches(_app())


def test_precision_matches_any_of():
    app_filter = AppFilter(precisions=frozenset({"w8a8", "float"}))
    assert app_filter.matches(_app(precisions=["w8a8"]))


def test_precision_no_match():
    app_filter = AppFilter(precisions=frozenset({"w4a16"}))
    assert not app_filter.matches(_app(precisions=["float"]))


def test_model_matches_substring():
    app_filter = AppFilter(models=("llama",))
    assert app_filter.matches(_app(related_models=["llama_v3_2_3b_instruct"]))


def test_model_no_match():
    assert not AppFilter(models=("whisper",)).matches(_app(related_models=["yamnet"]))


def test_device_matches_leniently():
    app_filter = AppFilter(device="snapdragon 8 elite qrd")
    assert app_filter.matches(_app(supported_devices=["Snapdragon® 8 Elite QRD"]))


def test_device_no_match():
    app_filter = AppFilter(device="Snapdragon X Elite CRD")
    assert not app_filter.matches(_app(supported_devices=["Snapdragon 8 Elite QRD"]))


def test_device_matches_app_with_no_declared_devices():
    # Mirrors App._ensure_device_supported: no declared devices means
    # unrestricted, not "matches nothing".
    assert AppFilter(device="Snapdragon 8 Elite QRD").matches(_app())


def test_dimensions_are_anded():
    app_filter = AppFilter(
        app_types=frozenset({AppType.UBUNTU}),
        languages=frozenset({AppLanguage.KOTLIN}),
    )
    assert not app_filter.matches(_app())


def test_build_app_filter_defaults_to_empty():
    assert build_app_filter().is_empty()


def test_build_app_filter_resolves_app_types():
    app_filter = build_app_filter(app_type=["android", "UBUNTU"])
    assert app_filter.app_types == frozenset({AppType.ANDROID, AppType.UBUNTU})


def test_build_app_filter_resolves_languages():
    app_filter = build_app_filter(language=["python"])
    assert app_filter.languages == frozenset({AppLanguage.PYTHON})


def test_build_app_filter_accepts_cpp_alias():
    assert build_app_filter(language=["CPP"]).languages == frozenset({AppLanguage.CPP})


def test_build_app_filter_accepts_cpp_value():
    assert build_app_filter(language=["c++"]).languages == frozenset({AppLanguage.CPP})


def test_build_app_filter_rejects_unknown_app_type():
    with pytest.raises(InvalidArgumentError, match="Valid app types"):
        build_app_filter(app_type=["macos"])


def test_build_app_filter_rejects_unknown_language():
    with pytest.raises(InvalidArgumentError, match="Valid languages"):
        build_app_filter(language=["rust"])


def test_build_app_filter_error_echoes_original_casing():
    with pytest.raises(InvalidArgumentError, match="'Andriod'"):
        build_app_filter(app_type=["Andriod"])


def test_build_app_filter_normalizes_free_form_values():
    app_filter = build_app_filter(
        runtime=["Geniex_QAIRT"],
        domain=["Generative-AI"],
        use_case=["Object Detection"],
        precision=["FLOAT"],
        model=["Whisper_Base"],
    )
    assert app_filter.runtimes == frozenset({"geniex qairt"})
    assert app_filter.domains == frozenset({"generative ai"})
    assert app_filter.use_cases == frozenset({"object detection"})
    assert app_filter.precisions == frozenset({"float"})
    assert app_filter.models == ("whisper base",)


def test_build_app_filter_drops_blank_values():
    assert build_app_filter(runtime=["", "  "]).runtimes == frozenset()


def test_build_app_filter_passes_device_through():
    app_filter = build_app_filter(device="Snapdragon 8 Elite QRD")
    assert app_filter.device == "Snapdragon 8 Elite QRD"
