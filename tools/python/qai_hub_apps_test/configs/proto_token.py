# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""A validated, string-valued token type usable as a Pydantic field.

``ProtoToken`` is a ``str`` subclass that normalizes and validates its input
via a subclass-provided rule, raising a ``ValueError`` on unknown values.
Instances *are* their token string, so they print, join, compare, hash, and
serialize as that string and work anywhere a ``str`` is accepted; a ``.value``
property is also provided for parity with enum-style access.

Members can also be referenced by name like an enum — e.g. ``Runtime.ONNX``
resolves to ``Runtime("onnx")`` — since member lookup runs through the same
validator (the underlying proto helpers accept case-insensitive names).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class _ProtoTokenMeta(type):
    """Resolve ``<Class>.<NAME>`` enum-style access through the validator."""

    def __getattr__(cls, name: str) -> ProtoToken:
        if name.startswith("_"):
            raise AttributeError(name)
        validate = cast("Callable[[Any], ProtoToken]", cls._validate)
        try:
            return validate(name)
        except ValueError as e:
            raise AttributeError(name) from e


class ProtoToken(str, metaclass=_ProtoTokenMeta):
    """A ``str`` subclass holding a validated, normalized token.

    Subclasses override ``_normalize`` to validate an input and return the
    canonical token string.
    """

    __slots__ = ()

    @staticmethod
    def _normalize(value: Any) -> str:
        """Return the canonical token for *value*, or raise ``KeyError``."""
        raise NotImplementedError

    @property
    def value(self) -> str:
        """The token string (mirrors enum-style ``.value`` access)."""
        return str(self)

    @classmethod
    def _validate(cls, value: Any) -> ProtoToken:
        try:
            return cls(cls._normalize(value))
        except KeyError as e:
            # KeyError messages from the proto helpers already list valid values.
            raise ValueError(str(e)) from e

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )
