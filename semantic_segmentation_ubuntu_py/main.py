# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import contextlib
import math
import queue
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import gi
import numpy as np
import qai_hub_apps_utils.webui as ui
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.fps import FpsCounter
from qai_hub_apps_utils.image_processing import resize_pad
from qai_hub_apps_utils.quantization import dequantize, quantize
from utils.input_processing import get_gstreamer_input_pipeline
from utils.model_io_processing import blend_mask, decode_mask
from utils.model_metadata import load_model_metadata

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

outq: queue.Queue[np.ndarray[Any, np.dtype[np.uint8]]] = queue.Queue(maxsize=4)


def on_new_sample(sink: Any) -> Any:
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps().get_structure(0)
    w, h = caps.get_value("width"), caps.get_value("height")

    # Map buffer memory as read-only
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK
    try:
        rowstride = mapinfo.size // h
        arr = np.frombuffer(mapinfo.data, dtype=np.uint8, count=h * rowstride)
        arr = arr.reshape(h, rowstride)[:, : w * 3].copy()
        arr = arr.reshape((h, w, 3))
    finally:
        buf.unmap(mapinfo)

    with contextlib.suppress(queue.Full):
        outq.put_nowait(arr)

    return Gst.FlowReturn.OK


def _set_input(
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    rgb_input: np.ndarray,
) -> None:
    """Quantize (if needed) and feed the preprocessed RGB input into the model.

    Parameters
    ----------
    interpreter
        TFLite interpreter for the segmentation model.
    input_details
        Input tensor details from interpreter.get_input_details().
    rgb_input
        Preprocessed RGB image of shape [1, H, W, 3], dtype uint8 in range [0, 255].
    """
    detail = input_details[0]
    dtype = detail["dtype"]
    scales = detail["quantization_parameters"]["scales"]
    zero_points = detail["quantization_parameters"]["zero_points"]
    if np.issubdtype(dtype, np.integer):
        # Per-tensor case: fuse the [0, 1] normalize and affine quantize into one
        # float32 pass (shared quantize() uses float64); fall back to it otherwise.
        if np.size(scales) == 1:
            inv = np.float32(1.0 / (255.0 * float(scales[0])))
            info = np.iinfo(dtype)
            q = np.rint(rgb_input.astype(np.float32) * inv) + int(zero_points[0])
            input_val = np.clip(q, info.min, info.max).astype(dtype)
        else:
            input_val = quantize(
                rgb_input.astype(np.float32) / 255.0,
                zero_points=zero_points,
                scales=scales,
            )
    else:
        # Float model expects RGB in [0, 1].
        input_val = (rgb_input.astype(np.float32) / 255.0).astype(dtype)
    interpreter.set_tensor(detail["index"], input_val)


def _get_output(
    interpreter: Interpreter,
    detail: dict[str, Any],
) -> np.ndarray:
    """Read one output tensor, dequantizing it if the model is quantized.

    Parameters
    ----------
    interpreter
        TFLite interpreter for the segmentation model.
    detail
        A single entry from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Output tensor as float, shape [1, H, W, num_classes] (NHWC) or
        [1, num_classes, H, W] (NCHW).
    """
    tensor = interpreter.get_tensor(detail["index"])
    scales = detail["quantization_parameters"]["scales"]
    # Only dequantize a genuinely quantized tensor. The VOC DeepLab/FCN models
    # emit the class-id mask as a plain uint8 tensor with EMPTY quant params
    # (scales has size 0); dequantizing it would try to broadcast against a (0,)
    # array and crash. An integer tensor with no scales is already the final output.
    if np.issubdtype(detail["dtype"], np.integer) and np.size(scales) > 0:
        tensor = dequantize(
            tensor,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=scales,
        )
    return tensor


def run_inference(
    rgb_frame: np.ndarray,
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    output_details: list[dict[str, Any]],
    input_size: tuple[int, int],
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    """Run the segmentation model on a single RGB frame and decode the mask.

    The frame is letterboxed (aspect-preserving resize + zero-pad via the shared
    ``resize_pad``) into the model input, then the padding is cropped back off the
    decoded mask so the class map covers exactly the frame's content. Models are
    trained on aspect-preserved inputs; a plain stretch degrades the mask.

    Parameters
    ----------
    rgb_frame
        Input RGB image as a numpy array of shape [H, W, 3], dtype uint8.
    interpreter
        TFLite interpreter for the segmentation model.
    input_details
        Input tensor details from interpreter.get_input_details().
    output_details
        Output tensor details from interpreter.get_output_details().
    input_size
        Model input (height, width).
    timings
        Optional dict; when provided, the elapsed seconds for the ``preprocess``,
        ``inference`` and ``postprocess`` stages are added into it (see --profile).

    Returns
    -------
    np.ndarray
        Per-pixel class-id map covering the frame content, dtype uint8.
    """
    input_height, input_width = input_size
    src_h, src_w = rgb_frame.shape[:2]

    t0 = time.perf_counter()

    # Preprocess: aspect-preserving resize + centered zero-pad + quantize/set input.
    canvas, scale, (pad_left, pad_top) = resize_pad(
        rgb_frame, (input_height, input_width)
    )
    content_h, content_w = math.floor(src_h * scale), math.floor(src_w * scale)
    input_val = np.expand_dims(canvas, axis=0)
    _set_input(interpreter, input_details, input_val)

    t1 = time.perf_counter()

    # Model execution on the NPU.
    interpreter.invoke()

    t2 = time.perf_counter()

    # Postprocess: read/dequantize output, argmax-decode, crop the padding back off.
    model_output = _get_output(interpreter, output_details[0])
    class_map = decode_mask(model_output)

    # The mask may be at a different resolution than the model input, so scale the
    # content box from input space to mask space.
    mask_h, mask_w = class_map.shape
    sy, sx = mask_h / input_height, mask_w / input_width
    t, b = round(pad_top * sy), round((pad_top + content_h) * sy)
    left_x, right_x = round(pad_left * sx), round((pad_left + content_w) * sx)
    cropped = class_map[t:b, left_x:right_x]

    t3 = time.perf_counter()

    if timings is not None:
        timings["preprocess"] += t1 - t0
        timings["inference"] += t2 - t1
        timings["postprocess"] += t3 - t2

    return cropped


def _report_timings(timings: dict[str, float], frames: int) -> None:
    """Print per-frame average stage latencies and the implied pipeline FPS.

    Parameters
    ----------
    timings
        Accumulated seconds per stage (``preprocess``, ``inference``,
        ``postprocess``) over ``frames`` frames.
    frames
        Number of frames the timings were accumulated over.
    """
    pre = timings["preprocess"] / frames * 1e3
    inf = timings["inference"] / frames * 1e3
    post = timings["postprocess"] / frames * 1e3
    total = pre + inf + post
    fps = 1e3 / total if total > 0 else float("inf")
    print(
        f"[profile] avg over {frames} frames  "
        f"preprocess={pre:.2f}ms  inference={inf:.2f}ms  "
        f"postprocess={post:.2f}ms  total={total:.2f}ms  ({fps:.1f} FPS)",
        flush=True,
    )


def main(args: argparse.Namespace) -> None:
    if args.list_devices:
        subprocess.call(["v4l2-ctl", "--list-devices"])
        return

    Gst.init(None)

    if args.video_gstreamer_source:
        video_source = args.video_gstreamer_source
    else:
        video_source = f"v4l2src name=camsrc device={args.video_device}"
    pipeline = Gst.parse_launch(
        get_gstreamer_input_pipeline(
            video_source, (args.video_source_width, args.video_source_height)
        )
    )
    appsink = pipeline.get_by_name("appsink")
    if not appsink:
        raise RuntimeError("Could not find appsink element named 'appsink'")

    appsink.set_property("emit-signals", True)
    appsink.connect("new-sample", on_new_sample)

    # Read the model's file name and I/O shapes from metadata.json (shipped with
    # the asset)
    models_dir = Path(C.MODELS_DIR)
    metadata = load_model_metadata(models_dir)
    input_size = (metadata.input_height, metadata.input_width)

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

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(
        "--------------------------- Gstreamer ----------------------------", flush=True
    )
    pipeline.set_state(Gst.State.PLAYING)
    fps_counter = FpsCounter()

    warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

    print(
        "--------------------------- Web server ----------------------------",
        flush=True,
    )

    # Per-stage profiling (opt-in via --profile). Accumulated over a window of
    # frames, then reported as per-frame averages so the preprocess / model /
    # postprocess split can be compared against the model's reported throughput.
    timings = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    profiled_frames = 0

    try:
        ui.start_thread()
        while True:
            rgb_frame = outq.get(timeout=5)

            class_map = run_inference(
                rgb_frame,
                interpreter,
                input_details,
                output_details,
                input_size,
                timings=timings if args.profile else None,
            )

            blend_start = time.perf_counter()
            blend_mask(rgb_frame, class_map)
            if args.profile:
                # Blending the overlay onto the frame is part of postprocessing.
                timings["postprocess"] += time.perf_counter() - blend_start
                profiled_frames += 1
                if profiled_frames >= args.profile_window:
                    _report_timings(timings, profiled_frames)
                    timings = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
                    profiled_frames = 0

            fps_counter.tick()

            ui.set_frame(rgb_frame[..., ::-1])

    except queue.Empty:
        print("Timed out waiting for input! Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Segmentation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list-devices", action="store_true", help="List options for --video-device"
    )
    group.add_argument(
        "--video-device",
        type=str,
        help='GStreamer v4l2src video device (e.g. "/dev/video0")',
    )
    group.add_argument(
        "--video-gstreamer-source",
        type=str,
        help='GStreamer video source (e.g. "v4l2src device=/dev/video2" or "qtiqmmfsrc name=camsrc camera=0")',
    )
    parser.add_argument(
        "--video-source-width",
        type=int,
        required=False,
        default=1024,
        help="Video width (input), default 1024",
    )
    parser.add_argument(
        "--video-source-height",
        type=int,
        required=False,
        default=768,
        help="Video height (input), default 768",
    )
    parser.add_argument(
        "--qairt-path",
        type=Path,
        required=True,
        help="Path to QAIRT SDK root",
    )
    parser.add_argument(
        "--hexagon-version",
        type=str,
        default="v73",
        help="Hexagon version of the device, e.g. v73, default v73",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-frame preprocess/inference/postprocess latencies",
    )
    parser.add_argument(
        "--profile-window",
        type=int,
        default=60,
        help="Frames to average per --profile report, default 60",
    )

    args = parser.parse_args()
    main(args)
