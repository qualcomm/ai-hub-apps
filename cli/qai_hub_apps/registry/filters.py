# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from qai_hub_models_cli.proto_helpers.platform import normalize_hw_name
from qai_hub_models_cli.proto_helpers.platform_enums import normalize_label

from qai_hub_apps.configs.app_yaml import AppLanguage, AppType
from qai_hub_apps.errors import InvalidArgumentError

if TYPE_CHECKING:
    from qai_hub_apps.registry.base import App

_EnumT = TypeVar("_EnumT", bound=Enum)

# Extra spellings accepted for enum filter values, keyed by normalized input.
_LANGUAGE_ALIASES = {"cpp": AppLanguage.CPP}


def _norm(value: str) -> str:
    """Normalize a value for lenient, case-insensitive matching.

    Lowercases, treats ``_``/``-`` as spaces, and collapses whitespace, so
    ``"Generative AI"``, ``"generative_ai"`` and ``"GENERATIVE-AI"`` all match.
    """
    return " ".join(normalize_label(value).split())


def _norm_set(values: list[str] | None) -> frozenset[str]:
    """Normalize *values* into a set, dropping blanks."""
    return frozenset(_norm(v) for v in values or [] if _norm(v))


def _resolve_enum(
    values: list[str] | None,
    enum_cls: type[_EnumT],
    aliases: dict[str, _EnumT],
    label: str,
) -> frozenset[_EnumT]:
    """Resolve raw filter strings to members of *enum_cls*.

    Parameters
    ----------
    values
        Raw filter values as typed by the user.
    enum_cls
        The enum to resolve against, matched on its values.
    aliases
        Extra accepted spellings, keyed by normalized input.
    label
        Singular noun for the dimension, used in the error message.

    Returns
    -------
    frozenset[_EnumT]
        The resolved enum members, empty when *values* is empty.

    Raises
    ------
    InvalidArgumentError
        If a value matches neither an enum value nor an alias.
    """
    by_value = {_norm(member.value): member for member in enum_cls}
    resolved = set()
    for raw in values or []:
        member = by_value.get(_norm(raw)) or aliases.get(_norm(raw))
        if member is None:
            valid = ", ".join(str(m.value) for m in enum_cls)
            raise InvalidArgumentError(
                f"'{raw}' is not a known {label}. Valid {label}s: {valid}."
            )
        resolved.add(member)
    return frozenset(resolved)


@dataclass(frozen=True)
class AppFilter:
    """Criteria for selecting apps from a registry.

    Different dimensions are ANDed; values within one dimension are ORed. String
    matching is case-insensitive and treats ``_``/``-`` as spaces.
    """

    app_types: frozenset[AppType] = frozenset()
    languages: frozenset[AppLanguage] = frozenset()
    runtimes: frozenset[str] = frozenset()
    domains: frozenset[str] = frozenset()
    use_cases: frozenset[str] = frozenset()
    precisions: frozenset[str] = frozenset()
    models: tuple[str, ...] = ()
    device: str | None = None

    def is_empty(self) -> bool:
        """Return whether no criteria are set, i.e. every app matches."""
        return not (
            self.app_types
            or self.languages
            or self.runtimes
            or self.domains
            or self.use_cases
            or self.precisions
            or self.models
            or self.device
        )

    def matches(self, app: App) -> bool:
        """Return whether *app* satisfies every criterion."""
        if self.app_types and app.app_type not in self.app_types:
            return False
        if self.languages and not self.languages.intersection(app.languages):
            return False
        if self.runtimes and not self.runtimes.intersection(
            _norm(r) for r in app.runtime
        ):
            return False
        if self.domains and _norm(app.domain) not in self.domains:
            return False
        if self.use_cases and _norm(app.use_case) not in self.use_cases:
            return False
        if self.precisions and not self.precisions.intersection(
            _norm(p) for p in app.precisions
        ):
            return False
        # Model ids are long and versioned (llama_v3_2_3b_instruct), so '--model
        # llama' has to work; every other dimension is a curated vocabulary.
        if self.models and not any(
            needle in _norm(model)
            for needle in self.models
            for model in app.related_models
        ):
            return False
        # An app that declares no devices is unrestricted rather than matching
        # nothing, mirroring App._ensure_device_supported.
        if self.device is not None and app.supported_devices:
            known = {normalize_hw_name(d) for d in app.supported_devices}
            if normalize_hw_name(self.device) not in known:
                return False
        return True


def build_app_filter(
    *,
    app_type: list[str] | None = None,
    language: list[str] | None = None,
    runtime: list[str] | None = None,
    domain: list[str] | None = None,
    use_case: list[str] | None = None,
    precision: list[str] | None = None,
    model: list[str] | None = None,
    device: str | None = None,
) -> AppFilter:
    """Build an :class:`AppFilter` from raw CLI filter values.

    Only the closed enum dimensions are validated. An unknown runtime, domain,
    use case, precision or model simply matches nothing, so the caller can show
    the registry's real vocabulary instead of an error.

    Parameters
    ----------
    app_type
        App types to accept (see :class:`~qai_hub_apps.configs.app_yaml.AppType`).
    language
        Implementation languages to accept.
    runtime
        Inference runtimes to accept.
    domain
        Domains to accept.
    use_case
        Use cases to accept.
    precision
        Model precisions to accept.
    model
        Model id substrings to accept.
    device
        A canonical AI Hub device name, already resolved by the caller.

    Returns
    -------
    AppFilter
        The filter described by the given values.

    Raises
    ------
    InvalidArgumentError
        If an app type or language is not recognized.
    """
    return AppFilter(
        app_types=_resolve_enum(app_type, AppType, {}, "app type"),
        languages=_resolve_enum(language, AppLanguage, _LANGUAGE_ALIASES, "language"),
        runtimes=_norm_set(runtime),
        domains=_norm_set(domain),
        use_cases=_norm_set(use_case),
        precisions=_norm_set(precision),
        models=tuple(sorted(_norm_set(model))),
        device=device,
    )
