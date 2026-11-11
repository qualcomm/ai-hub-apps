# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import concurrent.futures
import contextlib
import queue
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import gi
import numpy as np
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.config import AppInfo
from qai_hub_apps_utils.fps import FpsCounter
from qai_hub_apps_utils.image_processing import resize_pad
from qai_hub_apps_utils.input_devices import get_default_video_device
from qai_hub_apps_utils.platform import get_current_device
from qai_hub_apps_utils.quantization import dequantize, quantize
from qai_hub_apps_utils.webui import WebUI
from utils.input_processing import get_gstreamer_input_pipeline
from utils.model_io_processing import (
    colorize_depth,
    decode_depth,
    image_input_layout,
    undo_resize_pad,
)
from utils.model_metadata import load_model_metadata

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

outq: queue.Queue[np.ndarray[Any, np.dtype[np.uint8]]] = queue.Queue(maxsize=1)


def on_new_sample(sink: Any) -> Any:
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps().get_structure(0)
    w, h = caps.get_value("width"), caps.get_value("height")

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

    with contextlib.suppress(queue.Empty):
        outq.get_nowait()
    with contextlib.suppress(queue.Full):
        outq.put_nowait(arr)

    return Gst.FlowReturn.OK


def _set_input(
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    rgb_input: np.ndarray,
    channels_first: bool,
) -> None:
    """Quantize (if needed) and feed the preprocessed RGB input into the model.

    Parameters
    ----------
    interpreter
        TFLite interpreter for the depth-estimation model.
    input_details
        Input tensor details from interpreter.get_input_details().
    rgb_input
        Preprocessed RGB image of shape [1, H, W, 3], dtype uint8 in range [0, 255].
    channels_first
        Whether the model expects NCHW input; if so the tensor is transposed
        from NHWC before being set.
    """
    detail = input_details[0]
    if np.issubdtype(detail["dtype"], np.integer):
        normalized = rgb_input.astype(np.float32) / 255.0
        input_val = quantize(
            normalized,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        input_val = (rgb_input.astype(np.float32) / 255.0).astype(detail["dtype"])
    if channels_first:
        input_val = np.transpose(input_val, (0, 3, 1, 2))
    interpreter.set_tensor(detail["index"], input_val)


def _get_output(
    interpreter: Interpreter,
    detail: dict[str, Any],
) -> np.ndarray:
    """Read one output tensor, dequantizing it if the model is quantized.

    Parameters
    ----------
    interpreter
        TFLite interpreter for the depth-estimation model.
    detail
        A single entry from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Output depth tensor as float, batch-first and 4-D (NCHW or NHWC).
    """
    tensor = interpreter.get_tensor(detail["index"])
    if np.issubdtype(detail["dtype"], np.integer):
        tensor = dequantize(
            tensor,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    return tensor


def run_inference(
    rgb_frame: np.ndarray,
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    output_details: list[dict[str, Any]],
    input_size: tuple[int, int],
    channels_first: bool,
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    """Run the depth model on a single RGB frame and return a frame-sized heatmap.

    Parameters
    ----------
    rgb_frame
        Input RGB image as a numpy array of shape [H, W, 3], dtype uint8.
    interpreter
        TFLite interpreter for the depth-estimation model.
    input_details
        Input tensor details from interpreter.get_input_details().
    output_details
        Output tensor details from interpreter.get_output_details().
    input_size
        Model input (height, width) to resize-and-pad the frame to.
    channels_first
        Whether the model input/output are laid out as NCHW.
    timings
        If given, filled with per-stage wall time in ms: ``preprocess``, ``model``
        (set input, invoke, read output) and ``postprocess``.

    Returns
    -------
    np.ndarray
        RGB depth heatmap of shape [H, W, 3], dtype uint8, at the frame resolution.
    """
    t0 = time.perf_counter()
    frame_h, frame_w = rgb_frame.shape[:2]

    input_val, scale, pad = resize_pad(rgb_frame, input_size)
    input_val = np.expand_dims(input_val, axis=0)
    t1 = time.perf_counter()

    _set_input(interpreter, input_details, input_val, channels_first)
    interpreter.invoke()
    raw_output = _get_output(interpreter, output_details[0])
    t2 = time.perf_counter()

    depth = decode_depth(raw_output)
    depth = undo_resize_pad(depth, (frame_w, frame_h), scale, pad)
    heatmap = colorize_depth(depth)
    if timings is not None:
        timings["preprocess"] = (t1 - t0) * 1000
        timings["model"] = (t2 - t1) * 1000
        timings["postprocess"] = (time.perf_counter() - t2) * 1000
    return heatmap


def build_interpreter(
    model_path: str, args: argparse.Namespace
) -> tuple[Interpreter, str]:
    """Create a TFLite interpreter, preferring the QNN HTP delegate.

    On-device the model runs on the Hexagon NPU via the QNN TFLite delegate.
    Some models (notably large float graphs) can fail to compile for the HTP on
    a given device; in that case the app falls back to CPU execution so it still
    runs, printing a clear warning. ``--cpu`` forces the CPU path directly.

    Parameters
    ----------
    model_path
        Path to the ``.tflite`` model file.
    args
        Parsed CLI arguments (uses ``cpu``, ``qairt_path`` and ``hexagon_version``).

    Returns
    -------
    tuple[Interpreter, str]
        An allocated TFLite interpreter, and the backend it runs on ("NPU" when
        delegated to the HTP, else "CPU").
    """
    if not args.cpu:
        delegate_path = (
            args.qairt_path
            / "lib"
            / "aarch64-oe-linux-gcc11.2"
            / "libQnnTFLiteDelegate.so"
        )
        try:
            delegate = Delegate(
                delegate_path,
                {
                    "backend_type": "htp",
                    "htp_performance_mode": "2",
                    "library_path": str(
                        args.qairt_path
                        / "lib"
                        / "aarch64-oe-linux-gcc11.2"
                        / "libQnnHtp.so"
                    ),
                    "skel_library_dir": str(
                        args.qairt_path
                        / "lib"
                        / f"hexagon-{args.hexagon_version}"
                        / "unsigned"
                    ),
                },
            )
            interpreter = Interpreter(model_path, experimental_delegates=[delegate])
            interpreter.allocate_tensors()
            print("Using QNN HTP delegate for inference.", flush=True)
            return interpreter, "NPU"
        except (RuntimeError, ValueError, OSError) as exc:
            print(
                "WARNING: could not run the model on the HTP "
                f"({type(exc).__name__}: {exc}).\n"
                "         Falling back to CPU execution, which is significantly "
                "slower. Double-check --hexagon-version matches this device, or "
                "re-run with --cpu to skip the HTP attempt entirely.",
                flush=True,
            )

    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    print("Using CPU for inference.", flush=True)
    return interpreter, "CPU"


def main(args: argparse.Namespace) -> None:
    if args.list_devices:
        subprocess.call(["v4l2-ctl", "--list-devices"])
        return

    if not args.cpu and not args.hexagon_version:
        raise SystemExit(
            "Unknown Hexagon version for this device. "
            "Pass it with --hexagon-version <e.g. v73>, or use --cpu to run "
            "without the HTP."
        )

    Gst.init(None)

    if args.video_gstreamer_source:
        video_source = args.video_gstreamer_source
    else:
        try:
            device_path = args.video_device or get_default_video_device()
        except RuntimeError as error:
            raise SystemExit(
                f"{error} Pass a camera with --video-device <path> (see "
                "--list-devices), or a full GStreamer source with "
                "--video-gstreamer-source."
            ) from error
        video_source = f"v4l2src name=camsrc device={device_path}"
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

    models_dir = Path(C.MODELS_DIR)
    metadata = load_model_metadata(models_dir)
    channels_first, input_height, input_width = image_input_layout(metadata.input_shape)
    input_size = (input_height, input_width)

    model_path = str(models_dir / metadata.model_filename)
    interpreter, backend = build_interpreter(model_path, args)

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
    info = AppInfo.load()
    model_section = {
        "title": "Model",
        "rows": [
            ("Model", metadata.model_id),
            ("Use case", info.use_case),
            ("Precision", ", ".join(info.precisions)),
            ("Backend", f"{info.runtime.upper()} ({backend})"),
        ],
    }

    def publish(rgb_frame: np.ndarray, heatmap: np.ndarray) -> None:
        ui.set_frame(rgb_frame[..., ::-1], window=0)
        ui.set_frame(heatmap[..., ::-1], window=1)

    frames = 0
    try:
        ui = WebUI()
        ui.set_windows(1, 2, titles=["Live", "Depth"])
        ui.set_title(info.name)
        ui.start_thread()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as encoder:
            encode_future: concurrent.futures.Future | None = None
            while True:
                rgb_frame = outq.get(timeout=5)

                timings: dict[str, float] = {}
                heatmap = run_inference(
                    rgb_frame,
                    interpreter,
                    input_details,
                    output_details,
                    input_size,
                    channels_first,
                    timings,
                )

                fps_counter.tick()

                # Encode off the inference loop; drop the pair if the encoder is busy.
                if encode_future is None or encode_future.done():
                    encode_future = encoder.submit(publish, rgb_frame, heatmap)
                frames += 1
                if frames in (1, args.auto_capture_after_frames):
                    print(
                        f"Frame {frames} ({rgb_frame.shape[1]}x{rgb_frame.shape[0]}): "
                        + ", ".join(f"{k} {v:.1f} ms" for k, v in timings.items()),
                        flush=True,
                    )
                if frames == args.auto_capture_after_frames:
                    if args.output_image:
                        cv2.imwrite(str(args.output_image), heatmap[..., ::-1])
                    print("Auto-capture done. Exiting...", flush=True)
                    break
                ui.set_sections(
                    [
                        model_section,
                        {
                            "title": "Depth",
                            "rows": [("FPS", f"{fps_counter.fps():.1f}")],
                        },
                    ]
                )

    except queue.Empty:
        print("Timed out waiting for input! Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Estimation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list-devices", action="store_true", help="List options for --video-device"
    )
    group.add_argument(
        "--video-device",
        type=str,
        help='GStreamer v4l2src video device (e.g. "/dev/video0"). Defaults to '
        "the host's first working camera.",
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
    device = get_current_device()
    parser.add_argument(
        "--hexagon-version",
        type=str,
        default=device.htp_version if device and device.htp_version else None,
        help="Hexagon version of the device, e.g. v73. Defaults to the "
        "configured target device.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on the CPU instead of the HTP (slower). Useful when "
        "a model cannot be compiled for the device's HTP.",
    )

    parser.add_argument(
        "--auto-capture-after-frames",
        type=int,
        default=None,
        help="If set, save the depth heatmap of this frame to --output-image and "
        "exit. Used by test.sh to exercise the app headlessly.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Where --auto-capture-after-frames saves the depth heatmap.",
    )

    args = parser.parse_args()
    main(args)
