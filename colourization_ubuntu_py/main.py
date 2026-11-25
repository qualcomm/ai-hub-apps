# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
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
from qai_hub_apps_utils.input_devices import get_default_video_device
from qai_hub_apps_utils.platform import get_current_device
from qai_hub_apps_utils.quantization import dequantize, quantize
from qai_hub_apps_utils.webui import WebUI
from utils.input_processing import get_gstreamer_input_pipeline
from utils.model_io_processing import colorize, decode_chroma, preprocess
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

    # Keep only the freshest frame: discard any stale one before enqueuing.
    with contextlib.suppress(queue.Empty):
        while True:
            outq.get_nowait()
    with contextlib.suppress(queue.Full):
        outq.put_nowait(arr)

    return Gst.FlowReturn.OK


def _set_input(
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    normalized_input: np.ndarray,
    channels_first: bool,
) -> None:
    """Quantize (if needed) and feed the preprocessed input into the model.

    Parameters
    ----------
    interpreter
        TFLite interpreter for the colourization model.
    input_details
        Input tensor details from interpreter.get_input_details().
    normalized_input
        Preprocessed grayscale-RGB input of shape [1, H, W, 3], dtype float32 in
        range [0, 1].
    channels_first
        Whether the model expects NCHW input; if so the tensor is transposed
        from NHWC before being set.
    """
    detail = input_details[0]
    if np.issubdtype(detail["dtype"], np.integer):
        input_val = quantize(
            normalized_input,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        input_val = normalized_input.astype(detail["dtype"])
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
        TFLite interpreter for the colourization model.
    detail
        A single entry from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Output chroma tensor as float, batch-first and 4-D (NCHW or NHWC).
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
    """Run the colourization model on a single RGB frame and return a colourized frame.

    Parameters
    ----------
    rgb_frame
        Input RGB image as a numpy array of shape [H, W, 3], dtype uint8.
    interpreter
        TFLite interpreter for the colourization model.
    input_details
        Input tensor details from interpreter.get_input_details().
    output_details
        Output tensor details from interpreter.get_output_details().
    input_size
        Model input (height, width) to resize the grayscale image to.
    channels_first
        Whether the model input/output are laid out as NCHW.
    timings
        If given, filled with per-stage wall time in ms: ``preprocess``, ``model``
        (set input, invoke, read output) and ``postprocess``.

    Returns
    -------
    np.ndarray
        Colourized RGB image of shape [H, W, 3], dtype uint8, at the frame resolution.
    """
    t0 = time.perf_counter()
    frame_h, frame_w = rgb_frame.shape[:2]

    gray_rgb, orig_l = preprocess(rgb_frame, input_size)
    input_val = np.expand_dims(gray_rgb, axis=0)
    t1 = time.perf_counter()

    _set_input(interpreter, input_details, input_val, channels_first)
    interpreter.invoke()
    raw_output = _get_output(interpreter, output_details[0])
    t2 = time.perf_counter()

    chroma = decode_chroma(raw_output)
    result = colorize(chroma, orig_l, (frame_w, frame_h))
    if timings is not None:
        timings["preprocess"] = (t1 - t0) * 1000
        timings["model"] = (t2 - t1) * 1000
        timings["postprocess"] = (time.perf_counter() - t2) * 1000
    return result


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
    interpreter : Interpreter
        An allocated TFLite interpreter, delegated to the HTP when available.
    backend : str
        ``"NPU"`` if the HTP delegate is in use, else ``"CPU"``.
    """
    if not args.cpu:
        lib_dir = args.qairt_path / "lib"
        try:
            delegate = Delegate(
                lib_dir / "aarch64-oe-linux-gcc11.2" / "libQnnTFLiteDelegate.so",
                {
                    "backend_type": "htp",
                    "htp_performance_mode": "2",
                    "library_path": str(
                        lib_dir / "aarch64-oe-linux-gcc11.2" / "libQnnHtp.so"
                    ),
                    "skel_library_dir": str(
                        lib_dir / f"hexagon-{args.hexagon_version}" / "unsigned"
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
                "slower. Re-run with --cpu to skip the HTP attempt.",
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

    if not args.hexagon_version:
        raise SystemExit(
            "Unknown Hexagon version for this device. "
            "Pass it with --hexagon-version <e.g. v73>."
        )

    Gst.init(None)

    if args.video_gstreamer_source:
        video_source = args.video_gstreamer_source
    else:
        try:
            device = args.video_device or get_default_video_device()
        except RuntimeError as error:
            raise SystemExit(
                f"{error} Pass a camera with --video-device <path> (see "
                "--list-devices), or a full GStreamer source with "
                "--video-gstreamer-source."
            ) from error
        video_source = f"v4l2src name=camsrc device={device}"
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
    input_size = (metadata.input_height, metadata.input_width)

    model_path = str(models_dir / metadata.model_filename)
    interpreter, backend = build_interpreter(model_path, args)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(
        "--------------------------- Gstreamer ----------------------------", flush=True
    )
    pipeline.set_state(Gst.State.PLAYING)

    warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

    print(
        "--------------------------- Web server ----------------------------",
        flush=True,
    )
    info = AppInfo.load()
    model_rows = [
        ("Model", metadata.model_id),
        ("Use case", info.use_case),
        ("Precision", ", ".join(info.precisions)),
        ("Backend", f"{info.runtime.upper()} ({backend})"),
    ]
    ui = WebUI()

    # The first run pays one-time OpenCV/NumPy setup (~130 ms); do it before serving.
    run_inference(
        np.zeros((args.video_source_height, args.video_source_width, 3), np.uint8),
        interpreter,
        input_details,
        output_details,
        input_size,
        metadata.channels_first,
    )

    fps_counter = FpsCounter()
    _last_latency_ms = _latest_frame = None

    def info_sections() -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = [
            {"title": "Model", "rows": model_rows},
            {"title": "Live", "rows": [("FPS", f"{fps_counter.fps():.1f}")]},
        ]
        if _last_latency_ms is not None:
            sections.append(
                {
                    "title": "Last capture",
                    "rows": [("Inference latency", f"{_last_latency_ms:.1f} ms")],
                }
            )
        return sections

    def do_capture(frame: np.ndarray | None) -> None:
        """Colourize the latest frame into the "Colourized" window."""
        nonlocal _last_latency_ms
        if frame is None:
            print("Capture requested but no frame received yet.", flush=True)
            return
        print("Capture requested: colourizing the latest frame...", flush=True)
        start = time.perf_counter()
        timings: dict[str, float] = {}
        colorized = run_inference(
            frame,
            interpreter,
            input_details,
            output_details,
            input_size,
            metadata.channels_first,
            timings,
        )
        _last_latency_ms = (time.perf_counter() - start) * 1000
        ui.set_image(colorized[..., ::-1], window=1, max_width=None)
        ui.set_text(
            f"Captured in {_last_latency_ms:.0f} ms. Press 's' to re-capture.",
            window=1,
        )
        if args.output_image:
            cv2.imwrite(str(args.output_image), colorized[..., ::-1])
        print(
            f"Captured frame in {_last_latency_ms:.1f} ms "
            f"({frame.shape[1]}x{frame.shape[0]}; "
            + ", ".join(f"{k} {v:.1f} ms" for k, v in timings.items())
            + ").",
            flush=True,
        )

    ui.set_windows(1, 2, titles=["Live", "Colourized"])
    ui.set_title(info.name)
    ui.set_text("Press 's' to colourize the current frame.", window=1)
    ui.on_keypress("s", lambda: do_capture(_latest_frame))
    ui.on_keypress("S", lambda: do_capture(_latest_frame))

    frames_until_capture = args.auto_capture_after_frames
    last_sections_update = 0.0
    try:
        ui.start_thread()
        while True:
            rgb_frame = outq.get(timeout=5)
            _latest_frame = rgb_frame
            fps_counter.tick()
            ui.set_frame(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY), window=0)
            now = time.monotonic()
            if now - last_sections_update >= 0.5:
                ui.set_sections(info_sections())
                last_sections_update = now

            if frames_until_capture is not None:
                frames_until_capture -= 1
                if frames_until_capture == 0:
                    do_capture(_latest_frame)
                    print("Auto-capture done. Exiting...", flush=True)
                    break

    except queue.Empty:
        print("Timed out waiting for input! Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Colourization")
    group = parser.add_mutually_exclusive_group()
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
        help="If set, trigger a capture (as if 's' was pressed) after this many "
        "frames. Used by test.sh to exercise the capture flow headlessly.",
    )

    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="If set, save the colourized image from each capture to this path.",
    )

    args = parser.parse_args()
    main(args)
