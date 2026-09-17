# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import contextlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import qai_hub_apps_utils.webui as ui
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.platform import get_current_device
from qai_hub_apps_utils.quantization import dequantize, quantize
from utils.draw import build_legend, colorize_range_view, compose_view, render_bev
from utils.input_processing import load_scan, project_scan
from utils.model_io_processing import decode_class_map, unproject_labels
from utils.model_metadata import ModelMetadata, load_model_metadata


def _set_input(
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    model_input: np.ndarray,
) -> None:
    """Quantize (if needed) and feed the normalized projection into the model.

    Parameters
    ----------
    interpreter
        TFLite interpreter for the segmentation model.
    input_details
        Input tensor details from interpreter.get_input_details().
    model_input
        Normalized projection matching the model's input shape, float32.
    """
    detail = input_details[0]
    if np.issubdtype(detail["dtype"], np.integer):
        input_val = quantize(
            model_input,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        input_val = model_input.astype(detail["dtype"])
    interpreter.set_tensor(detail["index"], input_val)


def _get_output(interpreter: Interpreter, detail: dict[str, Any]) -> np.ndarray:
    """Read one output tensor, dequantizing it if the model is quantized.

    The float path returns a view into the interpreter's own buffer, which stays
    valid until the next invoke; the class map is decoded from it before then.

    Parameters
    ----------
    interpreter
        TFLite interpreter for the segmentation model.
    detail
        A single entry from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Per-class scores as float.
    """
    tensor = interpreter.tensor(detail["index"])()
    scales = detail["quantization_parameters"]["scales"]
    if np.issubdtype(detail["dtype"], np.integer) and np.size(scales) > 0:
        tensor = dequantize(
            tensor,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=scales,
        )
    return tensor


def run_inference(
    scan_path: Path,
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    output_details: list[dict[str, Any]],
    metadata: ModelMetadata,
    legend: np.ndarray,
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    """Segment one LiDAR scan and render it for display.

    Parameters
    ----------
    scan_path
        Path to the ``.bin`` scan to segment.
    interpreter
        TFLite interpreter for the segmentation model.
    input_details
        Input tensor details from interpreter.get_input_details().
    output_details
        Output tensor details from interpreter.get_output_details().
    metadata
        The model's I/O contract, giving the projection size and layout.
    legend
        Pre-rendered class-color key to place beside the bird's-eye view.
    timings
        Optional dict; when provided, the elapsed seconds for the ``preprocess``,
        ``inference`` and ``postprocess`` stages are added into it (see --profile).

    Returns
    -------
    np.ndarray
        RGB view of the segmented scan, dtype uint8.
    """
    t0 = time.perf_counter()

    points, remissions = load_scan(scan_path)
    projection = project_scan(
        points,
        remissions,
        metadata.input_height,
        metadata.input_width,
        metadata.channels_first,
    )
    _set_input(interpreter, input_details, projection.model_input)

    t1 = time.perf_counter()

    interpreter.invoke()

    t2 = time.perf_counter()

    class_map = decode_class_map(_get_output(interpreter, output_details[0]))
    point_labels = unproject_labels(class_map, projection)
    view = compose_view(
        colorize_range_view(class_map, projection.mask),
        render_bev(points, point_labels),
        legend,
    )

    t3 = time.perf_counter()

    if timings is not None:
        timings["preprocess"] += t1 - t0
        timings["inference"] += t2 - t1
        timings["postprocess"] += t3 - t2

    return view


def _report_timings(timings: dict[str, float]) -> None:
    """Print the latency of each pipeline stage.

    Parameters
    ----------
    timings
        Elapsed seconds per stage (``preprocess``, ``inference``,
        ``postprocess``).
    """
    stages = {stage: seconds * 1e3 for stage, seconds in timings.items()}
    print(
        "[profile] "
        + "  ".join(f"{stage}={ms:.2f}ms" for stage, ms in stages.items())
        + f"  total={sum(stages.values()):.2f}ms",
        flush=True,
    )


def main(args: argparse.Namespace) -> None:
    if not args.hexagon_version:
        raise SystemExit(
            "Unknown Hexagon version for this device. "
            "Pass it with --hexagon-version <e.g. v73>."
        )

    models_dir = Path(C.MODELS_DIR)
    metadata = load_model_metadata(models_dir)

    delegate_path = (
        args.qairt_path / "lib" / "aarch64-oe-linux-gcc11.2" / "libQnnTFLiteDelegate.so"
    )
    delegate = Delegate(
        delegate_path,
        {
            "backend_type": "htp",
            "htp_performance_mode": "2",
            "library_path": str(
                args.qairt_path / "lib" / "aarch64-oe-linux-gcc11.2" / "libQnnHtp.so"
            ),
            "skel_library_dir": str(
                args.qairt_path / "lib" / f"hexagon-{args.hexagon_version}" / "unsigned"
            ),
        },
    )

    interpreter = Interpreter(
        str(models_dir / metadata.model_filename), experimental_delegates=[delegate]
    )
    interpreter.allocate_tensors()

    legend = build_legend(C.BEV_SIZE_PX, C.RANGE_VIEW_WIDTH_PX - C.BEV_SIZE_PX)

    if args.output is None:
        print(
            "--------------------------- Web server ----------------------------",
            flush=True,
        )
        ui.start_thread()

    timings = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    view = run_inference(
        args.lidar_source,
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details(),
        metadata,
        legend,
        timings=timings if args.profile else None,
    )

    if args.profile:
        _report_timings(timings)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), view[..., ::-1])
        print(f"Wrote {args.output}", flush=True)
        return

    ui.set_frame(view[..., ::-1])
    print("Serving the segmented scan on port 8080.", flush=True)
    with contextlib.suppress(EOFError, KeyboardInterrupt):
        input("Press Enter to exit. ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiDAR Semantic Segmentation")
    parser.add_argument(
        "--lidar-source",
        type=Path,
        required=True,
        help="The KITTI-format .bin LiDAR scan to segment",
    )
    parser.add_argument(
        "--qairt-path",
        type=Path,
        required=True,
        help="Path to QAIRT SDK root",
    )
    device = get_current_device()
    parser.add_argument(
        "--hexagon-version",
        type=str,
        default=device.htp_version if device and device.htp_version else None,
        help="Hexagon version of the device, e.g. v73. Defaults to the "
        "configured target device.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the segmented view to this image path instead of serving it",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-scan preprocess/inference/postprocess latencies",
    )

    args = parser.parse_args()
    main(args)
