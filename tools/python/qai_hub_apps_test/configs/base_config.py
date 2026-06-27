# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Pydantic config base with YAML (de)serialization.

Provides ``from_yaml`` / ``to_yaml`` backed by ``pydantic_yaml`` + ``ruamel``
so emitted YAML (e.g. ``registry.yaml``) matches the canonical formatting:
defaults and ``None`` values omitted, no line wrapping, and multi-line strings
written as block scalars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ruamel.yaml
from pydantic import BaseModel, ConfigDict
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from ruamel.yaml.representer import RoundTripRepresenter
from typing_extensions import Self


class BaseConfig(BaseModel):
    """Pydantic v2 model with YAML load/dump helpers."""

    # Forbid unknown keys so malformed configs fail loudly.
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load an instance from the YAML file at *path*."""
        return parse_yaml_file_as(cls, path)

    def to_yaml(self, path: str | Path, write_if_empty: bool = False) -> bool:
        """Dump this instance to a YAML file.

        Defaults and ``None`` values are omitted. If the result is empty, the
        file is written only when *write_if_empty* is True, and an empty file
        is removed. Returns True if a non-empty file was written.
        """
        yaml = ruamel.yaml.YAML()
        # Avoid wrapping long strings onto multiple lines (simplistic readers).
        yaml.width = 4096

        # Dump strings containing newlines as block scalars rather than "\n".
        def _yaml_repr_str(dumper: RoundTripRepresenter, data: str) -> Any:
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.representer.add_representer(str, _yaml_repr_str)

        to_yaml_file(
            path,
            self,
            custom_yaml_writer=yaml,
            exclude_defaults=True,
            exclude_none=True,
        )

        if os.path.getsize(path) == 0:
            if not write_if_empty:
                os.remove(path)
            return False
        return True
