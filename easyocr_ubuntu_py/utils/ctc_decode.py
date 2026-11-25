# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Greedy CTC decoding for the recognizer output.

Adapted from EasyOCR v1.7.2 `easyocr/utils.py` (CTCLabelConverter.decode_greedy,
custom_mean), Copyright (c) JaidedAI (Apache-2.0).
"""

from __future__ import annotations

import numpy as np


def custom_mean(x: np.ndarray) -> np.float64:
    """EasyOCR's confidence aggregation: geometric-mean-like score."""
    return x.prod() ** (2.0 / np.sqrt(len(x)))


class CTCGreedyDecoder:
    """Decode recognizer logits/probabilities into text using greedy CTC.

    The character set order MUST match the model's output class ordering. The
    CTC blank token is class 0; output class ``i`` (i >= 1) maps to
    ``charset[i - 1]``.
    """

    def __init__(self, charset: str) -> None:
        # Index 0 is the CTC blank; real characters start at index 1.
        self.character = ["[blank]", *list(charset)]

    def decode(self, probs: np.ndarray) -> tuple[str, np.float64]:
        """
        Parameters
        ----------
        probs
            Per-timestep class probabilities for a single crop, shape
            [T, num_classes]. Should already be softmax-normalized.

        Returns
        -------
        text : str
            Decoded string (repeats collapsed, blanks removed).
        confidence : np.float64
            Aggregated confidence in [0, 1].
        """
        pred_index = probs.argmax(axis=1)  # [T]

        # Collapse consecutive duplicates, then drop blanks (index 0).
        not_repeat = np.insert(pred_index[1:] != pred_index[:-1], 0, True)
        not_blank = pred_index != 0
        keep = not_repeat & not_blank
        text = "".join(np.array(self.character)[pred_index[keep]])

        # Confidence: product of per-timestep max prob over non-blank timesteps.
        max_probs = probs.max(axis=1)[pred_index != 0]
        if len(max_probs) == 0:
            max_probs = np.array([0.0])
        confidence = custom_mean(max_probs)

        return text, confidence
