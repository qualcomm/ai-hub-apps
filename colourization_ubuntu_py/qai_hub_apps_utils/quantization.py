# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np


def dequantize(
    values: np.ndarray, zero_points: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    return np.asarray(
        ((np.int32(values) - np.int32(zero_points)) * np.float64(scales)),
        dtype=np.float32,
    )


def quantize(
    values: np.ndarray, zero_points: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    z = np.asarray(zero_points, dtype=np.int32)
    s = np.asarray(scales, dtype=np.float64)

    q_float = np.rint(v / s) + z

    info = np.iinfo(np.uint8)
    q_clipped = np.clip(q_float, info.min, info.max)

    return q_clipped.astype(np.uint8, copy=False)
