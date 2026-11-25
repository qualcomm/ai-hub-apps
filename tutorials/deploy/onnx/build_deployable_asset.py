# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Tensor


def convert_initializers_int8_to_int4(
    model: onnx.ModelProto, initializers_convert_to_int4: list[str]
):
    """
    Convert INT8 (used as placeholder) initializers to INT4.
    """
    for name in initializers_convert_to_int4:
        inits = [x for x in model.graph.initializer if x.name == name]
        assert inits, f"Entry in initializers_convert_to_int4 not found: {name}"
        assert (
            inits[0].data_type == onnx.TensorProto.INT8
        ), f"Entry in initializers_convert_to_int4 must be INT8, found: {inits[0].data_type} (for {name})"
        # convert INT8 to INT4
        x = np.frombuffer(inits[0].raw_data, dtype=np.int8)
        x = np.concatenate([x, np.zeros(x.size % 2)])
        inits[0].data_type = onnx.TensorProto.INT4
        new_raw_data = (
            ((x[1::2].astype(np.uint8) & 0b1111) << 4)
            | (x[::2].astype(np.uint8) & 0b1111)
        ).tobytes()
        inits[0].raw_data = new_raw_data


def _natively_quantize_tensor(
    variable: Tensor,
    scales: np.ndarray,
    zero_points: np.ndarray,
    dtype: np.dtype,
    is_int4: bool,
    axis: int,
    name: str,
):
    """Quantize the weights/bias to store correctly in onnx graph's node."""
    rank = len(variable.shape)
    if is_int4:
        mn, mx = -8, 7
    else:
        iinfo = np.iinfo(dtype)
        mn, mx = iinfo.min, iinfo.max

    if rank == 0:
        w_unclipped = (
            np.round(
                np.array(variable.values).astype(np.float64) / scales.astype(np.float64)
            ).astype(np.int64)
            + zero_points
        )
        return np.array(w_unclipped.clip(mn, mx).astype(dtype))
    axis = axis % rank
    trailing_ones = rank - axis - 1
    broadcast_shape = (-1,) + (1,) * trailing_ones
    scales_fullrank = scales.reshape(broadcast_shape)
    zero_points_fullrank = zero_points.reshape(broadcast_shape)
    w_unclipped = (
        np.round(
            variable.values.astype(np.float64) / scales_fullrank.astype(np.float64)
        ).astype(np.int64)
        + zero_points_fullrank
    )

    w_unclipped_min = w_unclipped.min()
    w_unclipped_max = w_unclipped.max()
    if w_unclipped_min < mn or mx < w_unclipped_max:
        print(
            f"Parameter {name} in quantized range [{w_unclipped.min()}, {w_unclipped.max()}] is being clipped to [{mn}, {mx}]; this may result in accuracy loss."
        )

    w = w_unclipped.clip(mn, mx).astype(dtype)
    return w


def _convert_to_deployable_onnx(
    onnx_filepath: str,
):
    """
    Converts an ONNX graph (QDQ format with weights are stored as float 32) to ONNX graph where all the quantization is applied.
    The nodes in input graph look like:
        Constant weights (FP) -> Q -> DQ
    The nodes in output graph:
        Constant weights (INT) -> DQ
    """

    onnx_model = onnx.load(onnx_filepath)
    graph = gs.import_onnx(onnx_model)
    b_quantized_dtype = np.dtype("int32")
    unquantized_dtype = np.dtype("float32")
    initializers_convert_to_int4 = []
    for node in graph.nodes:
        bias_present = (
            node.op in ["Gemm", "Conv", "TransposeConv"] and len(node.inputs) == 3
        )
        for inputs in node.inputs:
            for dequantized_node in inputs.inputs:
                if (
                    dequantized_node.op == "DequantizeLinear"
                    and len(dequantized_node.inputs[0].inputs) > 0
                    and dequantized_node.i(0, 0).op == "QuantizeLinear"
                    and isinstance(dequantized_node.i(0, 0).inputs[0], gs.Constant)
                ):

                    quantize_node = dequantized_node.i(0, 0)

                    scales_w = quantize_node.inputs[1].values
                    np_dtype = dequantized_node.inputs[2].dtype

                    # Detect if type is INT4. Graph surgeon does not support
                    # INT4 well, so we have to store these as INT8 and then
                    # do a touch-up pass directly on the proto format.
                    is_int4 = dequantized_node.inputs[2].values.dtype.names == ("int4",)

                    zero_points_w = np.array(
                        quantize_node.inputs[2].values,
                        dtype=np_dtype,
                    )

                    weights_data = quantize_node.inputs[0]

                    # Quantize the weight initializer
                    quantized_weight = _natively_quantize_tensor(
                        weights_data,
                        scales_w,
                        zero_points_w,
                        dtype=np_dtype,
                        is_int4=is_int4,
                        axis=dequantized_node.attrs.get("axis", 0),
                        name=node.name,
                    )

                    # Add new information to dequantize linear
                    if is_int4:
                        # Graph surgeon will fail unless we make sure these
                        # types are INT8 here
                        dequantized_node.inputs[2].values = dequantized_node.inputs[
                            2
                        ].values.astype(np.int8)
                        dequantized_node.inputs[0] = gs.Constant(
                            dequantized_node.inputs[0].name + "_folded",
                            quantized_weight.astype(np.int8),
                        )
                        initializers_convert_to_int4.append(
                            dequantized_node.inputs[0].name
                        )
                        initializers_convert_to_int4.append(
                            dequantized_node.inputs[2].name
                        )
                    else:
                        dequantized_node.inputs[0] = gs.Constant(
                            dequantized_node.inputs[0].name + "_folded",
                            quantized_weight,
                        )

                    # Only applicable when bias is present
                    if (
                        bias_present
                        and node.inputs[0].inputs[0].op == "DequantizeLinear"
                        and len(node.inputs[0].inputs[0].inputs) > 2
                    ):
                        bias_var = node.inputs[2]

                        # Scale of bias is product of scale of weights and inputs
                        input_scales = node.i(0, 0).inputs[1].values

                        scales_b = np.array(input_scales * scales_w)
                        zero_points_b = np.full(scales_b.shape, 0).astype(
                            b_quantized_dtype
                        )
                        # Quantize the bias
                        quantized_bias = _natively_quantize_tensor(
                            bias_var,
                            scales_b,
                            zero_points_b,
                            dtype=b_quantized_dtype,
                            is_int4=False,
                            axis=node.attrs.get("axis", 0),
                            name=node.name,
                        )
                        dq_node_name = dequantized_node.inputs[0].name + "_b_folded"
                        d_bias = gs.Constant(dq_node_name, quantized_bias)

                        # Add dequantize linear node
                        dq_b_scales = gs.Constant(bias_var.name + "_scales", scales_b)
                        dq_b_zero_points = gs.Constant(
                            bias_var.name + "_zero_points", zero_points_b
                        )
                        dq_y = gs.Variable(
                            bias_var.name + "_dq",
                            dtype=unquantized_dtype,
                            shape=bias_var.shape,
                        )

                        node.inputs[2] = dq_y
                        dq = gs.Node(
                            op="DequantizeLinear",
                            inputs=[d_bias, dq_b_scales, dq_b_zero_points],
                            outputs=[dq_y],
                            attrs=dequantized_node.attrs,
                        )
                        graph.nodes.append(dq)

    # Graph cleanup removes all dangling nodes
    # Topological sort is needed by ORT
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)

    convert_initializers_int8_to_int4(model, initializers_convert_to_int4)

    # Save the modified graph in filepath.
    try:
        onnx.save_model(model, onnx_filepath)
    except ValueError as e:
        assert "The proto size is larger than the 2 GB limit" in str(e)
        onnx.save_model(
            model,
            onnx_filepath,
            save_as_external_data=True,
            location="model.data",
        )


def convert_to_deployable_onnx(
    onnx_filepath: str,
):
    try:
        _convert_to_deployable_onnx(
            onnx_filepath,
        )
    except Exception as e:
        # if anything fails, leave the model as is.
        # The ONNX model can be of unknown provenance and can fail for many reasons. Catching any exception and
        # leaving the model unchanged avoids having to deal with these failures.
        raise RuntimeError(
            f"convert_to_deployable_onnx: converting the ONNX model resulted in the following exception: {str(e)}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Convert to deployable ONNX asset (QDQ that maps to QOp representation)"
    )
    parser.add_argument("-f", "--onnx_file", required=True)
    args = parser.parse_args()
    convert_to_deployable_onnx(args.onnx_file)
