# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""The environment and consent prompts shared with the apps' generated scripts.

Mirrors require_consent in apps/_shared/scripts/interactive.{sh,ps1}.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Environment variable the app scripts read to skip their consent prompts.
# See require_consent in apps/_shared/scripts/interactive.{sh,ps1}.
NON_INTERACTIVE_ENV_VAR = "NON_INTERACTIVE"


# Set once from main() when --yes is passed: assume "yes" for every prompt,
# both the CLI's own confirm() calls and the app scripts' consent prompts.
NON_INTERACTIVE = False


def set_non_interactive(value: bool = True) -> None:
    """Assume "yes" for all prompts from here on."""
    global NON_INTERACTIVE  # noqa: PLW0603
    NON_INTERACTIVE = value
    logger.debug("Non-interactive mode: %s", value)


def is_non_interactive() -> bool:
    """Whether prompts should be auto-approved."""
    return NON_INTERACTIVE or os.environ.get(NON_INTERACTIVE_ENV_VAR) == "true"


def script_env() -> dict[str, str]:
    """Return the environment overrides for an app's generated script."""
    return {NON_INTERACTIVE_ENV_VAR: "true"} if is_non_interactive() else {}


def confirm(question: str, default: bool = False) -> bool:
    """Ask *question* as a y/N prompt.

    Parameters
    ----------
    question
        The question to print, without the ``[y/N]`` suffix.
    default
        The answer to use when there is nothing to read from (no tty, or the
        user interrupted).

    Returns
    -------
    bool
        True if the user approved.
    """
    if is_non_interactive():
        logger.debug("Auto-approving: %s", question)
        return True
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        logger.debug("No answer available; using default %s", default)
        return default
    return answer in ("y", "yes")
