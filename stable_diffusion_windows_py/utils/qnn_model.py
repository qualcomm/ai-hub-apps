# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Run Stable Diffusion's quantized ONNX graphs on the QNN Execution Provider.

The text_encoder/unet/vae graphs wrap each quantized boundary tensor in a
DequantizeLinear/QuantizeLinear pair (feeding a precompiled EPContext node)
purely so the scale/zero_point can be recovered from the graph; the raw ONNX
input/output tensors are genuinely uint16. ``load_quantized_model`` parses those
nodes to recover each boundary's (scale, zero_point); ``run_quantized`` then
quantizes inputs / dequantizes outputs at the Python/ORT boundary and runs the
session. Non-quantized boundary tensors are cast to their declared dtype and
passed through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from qai_hub_apps_utils.onnxruntime_qnn import open_qnn_session

_ORT_TYPE_TO_NP_DTYPE: dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
}


@dataclass
class QuantizedModel:
    """A loaded QNN session plus the (scale, zero_point) for its quantized boundaries."""

    session: ort.InferenceSession
    input_names: list[str]
    input_dtypes: list[type]
    output_names: list[str]
    input_qdq: dict[str, tuple[float, int]]
    output_qdq: dict[str, tuple[float, int]]


def load_quantized_model(model_path: str) -> QuantizedModel:
    """Open ``model_path`` on the QNN EP and recover its quantization params.

    Parameters
    ----------
    model_path
        Path to the ONNX model.

    Returns
    -------
    QuantizedModel
        The session plus per-boundary (scale, zero_point) for quantized inputs/outputs.
    """
    session = open_qnn_session(model_path)
    input_names = [i.name for i in session.get_inputs()]
    input_dtypes = [_ORT_TYPE_TO_NP_DTYPE[i.type] for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    input_qdq, output_qdq = _extract_qdq_params(model_path, input_names, output_names)
    return QuantizedModel(
        session, input_names, input_dtypes, output_names, input_qdq, output_qdq
    )


def run_quantized(
    model: QuantizedModel, *args: np.ndarray
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Run ``model`` on positional numpy arrays (in the graph's input order).

    Quantized inputs are quantized to their uint dtype; quantized outputs are
    dequantized back to float32. Other tensors are cast to their declared dtype.

    Parameters
    ----------
    model
        A model returned by ``load_quantized_model``.
    *args
        Input arrays, one per graph input, in declared order.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, ...]
        The single output array, or a tuple of outputs for multi-output graphs.
    """
    feed: dict[str, np.ndarray] = {}
    for name, dtype, arg in zip(
        model.input_names, model.input_dtypes, args, strict=False
    ):
        if name in model.input_qdq:
            scale, zero_point = model.input_qdq[name]
            q = np.rint(arg.astype(np.float64) / scale) + zero_point
            info = np.iinfo(dtype)
            feed[name] = np.clip(q, info.min, info.max).astype(dtype)
        else:
            feed[name] = arg.astype(dtype, copy=False)

    outputs = model.session.run(model.output_names, feed)

    results = []
    for name, out in zip(model.output_names, outputs, strict=False):
        if name in model.output_qdq:
            scale, zero_point = model.output_qdq[name]
            out = ((out.astype(np.int32) - zero_point) * scale).astype(np.float32)
        results.append(out)
    return results[0] if len(results) == 1 else tuple(results)


def _extract_qdq_params(
    model_path: str, input_names: list[str], output_names: list[str]
) -> tuple[dict[str, tuple[float, int]], dict[str, tuple[float, int]]]:
    """Scan the graph for DequantizeLinear (on inputs) / QuantizeLinear (on outputs)
    nodes to recover each boundary tensor's (scale, zero_point).
    """
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(model_path, load_external_data=False).graph
    initializers = {
        init.name: numpy_helper.to_array(init) for init in graph.initializer
    }

    input_qdq: dict[str, tuple[float, int]] = {}
    output_qdq: dict[str, tuple[float, int]] = {}
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and node.input[0] in input_names:
            scale = float(initializers[node.input[1]])
            zero_point = int(initializers[node.input[2]]) if len(node.input) > 2 else 0
            input_qdq[node.input[0]] = (scale, zero_point)
        elif node.op_type == "QuantizeLinear" and node.output[0] in output_names:
            scale = float(initializers[node.input[1]])
            zero_point = int(initializers[node.input[2]]) if len(node.input) > 2 else 0
            output_qdq[node.output[0]] = (scale, zero_point)
    return input_qdq, output_qdq
