# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""On-device EasyOCR app for Ubuntu + QAIRT.

Reads frames from a GStreamer source and streams them live over a minimal web
UI (port 8080), which shows the live feed and the most recent OCR result
side by side. The two-stage OCR pipeline (QNN TFLite delegate on the HTP)
only runs on demand: press 's' in the browser to run detection + recognition
on the current frame; the captured frame with selectable/copyable text boxes
overlaid stays in the "Detected" window until the next capture.
"""

import argparse
import contextlib
import queue
import subprocess
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import gi
import numpy as np
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.config import AppInfo
from qai_hub_apps_utils.draw import draw_box_from_xyxy
from qai_hub_apps_utils.input_devices import get_default_video_device
from qai_hub_apps_utils.platform import get_current_device
from qai_hub_apps_utils.quantization import dequantize, quantize
from qai_hub_apps_utils.webui import WebUI
from utils.input_processing import get_gstreamer_input_pipeline
from utils.ocr_pipeline import OCRPipeline

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

# Model file locations, matching `model_file_paths` in info.yaml (populated by
# `qai-hub-apps fetch`).
DETECTOR_MODEL = "models/easyocr_detector.tflite"
RECOGNIZER_MODEL = "models/easyocr_recognizer.tflite"

outq: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)


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

    with contextlib.suppress(queue.Full):
        outq.put_nowait(arr)

    return Gst.FlowReturn.OK


def _nhwc_input_size(interpreter: Interpreter) -> tuple[int, int]:
    """Return (height, width) of an interpreter's single NHWC image input."""
    shape = interpreter.get_input_details()[0]["shape"]
    return int(shape[1]), int(shape[2])


def _unit_float(sample: np.ndarray) -> np.ndarray:
    """Normalize a sample to float32 in [0, 1] (uint8 samples are in [0, 255])."""
    if sample.dtype == np.uint8:
        return sample * np.float32(1.0 / 255.0)
    return sample.astype(np.float32, copy=False)


def _make_model_fn(interpreter: Interpreter) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a TFLite interpreter as a float-out callable.

    The returned callable accepts a single sample WITHOUT a batch dim, either
    uint8 in [0, 255] or float32 in [0, 1], handles (de)quantization per the
    model's tensor details, and returns the dequantized float output (batch dim
    preserved).
    """
    in_detail = interpreter.get_input_details()[0]
    in_qp = in_detail["quantization_parameters"]
    out_detail = interpreter.get_output_details()[0]
    out_qp = out_detail["quantization_parameters"]

    u8_input = in_detail["dtype"] == np.uint8 and np.size(in_qp["scales"]) == 1
    u8_table: np.ndarray | None = None
    if u8_input:
        table = quantize(
            np.arange(256, dtype=np.float32) / 255.0,
            zero_points=in_qp["zero_points"],
            scales=in_qp["scales"],
        )
        if not np.array_equal(table, np.arange(256, dtype=np.uint8)):
            u8_table = table

    def run(sample: np.ndarray) -> np.ndarray:
        batched = sample[np.newaxis]
        if u8_input and sample.dtype == np.uint8:
            value = batched if u8_table is None else u8_table[batched]
        elif np.issubdtype(in_detail["dtype"], np.integer):
            value = quantize(
                _unit_float(batched),
                zero_points=in_qp["zero_points"],
                scales=in_qp["scales"],
            )
        else:
            value = _unit_float(batched).astype(in_detail["dtype"], copy=False)
        interpreter.set_tensor(in_detail["index"], value)
        interpreter.invoke()

        tensor = interpreter.get_tensor(out_detail["index"])
        scales = out_qp["scales"]
        if np.issubdtype(out_detail["dtype"], np.integer) and np.size(scales) > 0:
            if np.size(scales) == 1:
                tensor = (
                    tensor.astype(np.float32) - np.float32(out_qp["zero_points"][0])
                ) * np.float32(scales[0])
            else:
                tensor = dequantize(
                    tensor, zero_points=out_qp["zero_points"], scales=scales
                )
        return tensor

    return run


def _load_interpreter(model_path: str, delegate: Delegate) -> tuple[Interpreter, bool]:
    """Load a TFLite model on the QNN HTP delegate, falling back to CPU.

    The QNN HTP delegate only supports static-shaped graphs. Some OCR graphs
    (notably EasyOCR's recognizer, whose LSTM/CTC path contains dynamic-sized
    tensors) are rejected by `ModifyGraphWithDelegate`; for those we run on CPU
    so the app still works end to end.
    Returns (interpreter, used_cpu_fallback) -- the fallback flag is surfaced
    in the "Info" window so it's clear when a model isn't actually running on
    the HTP.
    """
    try:
        interpreter = Interpreter(model_path, experimental_delegates=[delegate])
        interpreter.allocate_tensors()
        return interpreter, False
    except (RuntimeError, ValueError):
        warnings.warn(
            f"QNN HTP delegate rejected '{model_path}' (likely dynamic-shaped "
            "tensors); running this model on CPU instead.",
            stacklevel=2,
        )
        interpreter = Interpreter(model_path)
        interpreter.allocate_tensors()
        return interpreter, True


def _build_delegate(args: argparse.Namespace) -> Delegate:
    lib_dir = args.qairt_path / "lib" / "aarch64-oe-linux-gcc11.2"
    return Delegate(
        str(lib_dir / "libQnnTFLiteDelegate.so"),
        {
            "backend_type": "htp",
            "htp_performance_mode": "2",
            "library_path": str(lib_dir / "libQnnHtp.so"),
            "skel_library_dir": str(
                args.qairt_path / "lib" / f"hexagon-{args.hexagon_version}" / "unsigned"
            ),
        },
    )


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

    delegate = _build_delegate(args)
    detector, detector_on_cpu = _load_interpreter(DETECTOR_MODEL, delegate)
    # Interpreters aren't thread-safe: one recognizer per worker.
    recognizers = [
        _load_interpreter(RECOGNIZER_MODEL, delegate)
        for _ in range(max(1, args.recognizer_workers))
    ]
    recognizer_on_cpu = recognizers[0][1]
    recognizers = [interpreter for interpreter, _ in recognizers]

    ocr = OCRPipeline(
        detector_fn=_make_model_fn(detector),
        recognizer_fns=[_make_model_fn(r) for r in recognizers],
        detector_hw=_nhwc_input_size(detector),
        recognizer_hw=_nhwc_input_size(recognizers[0]),
        max_recognitions=args.max_recognitions,
    )

    print(
        "--------------------------- Gstreamer ----------------------------", flush=True
    )
    pipeline.set_state(Gst.State.PLAYING)
    warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

    info = AppInfo.load()
    runtime = info.runtime.upper()
    model_section = {
        "title": "Model",
        "rows": [
            ("Model", info.name),
            ("Use case", info.use_case),
            ("Precision", ", ".join(info.precisions)),
            (
                "Backend",
                [
                    ("Detector", f"{runtime} ({'CPU' if detector_on_cpu else 'NPU'})"),
                    (
                        "Recognizer",
                        f"{runtime} ({'CPU' if recognizer_on_cpu else 'NPU'})",
                    ),
                ],
            ),
        ],
    }

    ui = WebUI()

    def _info_sections() -> list[dict[str, Any]]:
        stats = ocr.stats
        return [
            model_section,
            {
                "title": "Last capture",
                "rows": [
                    ("Detector latency", f"{stats.detector_ms:.1f} ms"),
                    (
                        "Recognizer latency",
                        f"{stats.recognizer_ms_total:.1f} ms total, "
                        f"{stats.recognizer_ms_avg:.1f} ms/box avg",
                    ),
                    ("Total capture latency", f"{stats.total_ms:.1f} ms"),
                    ("Boxes detected", str(stats.num_boxes_detected)),
                    (
                        "Boxes recognized",
                        str(stats.num_boxes_recognized)
                        + (
                            f" (capped at {args.max_recognitions})"
                            if args.max_recognitions
                            else ""
                        ),
                    ),
                ],
            },
        ]

    def do_capture(frame: np.ndarray | None) -> None:
        """Run OCR on the latest frame and store the result."""
        if frame is None:
            print("Capture requested but no frame received yet.", flush=True)
            return

        detections = ocr.run(frame)
        annotated = frame.copy()
        for x0, y0, x1, y1, text, _conf in detections[: args.max_recognitions]:
            draw_box_from_xyxy(
                annotated, (x0, y0), (x1, y1), color=(0, 255, 0), size=2, text=text
            )
        ui.set_image(annotated[..., ::-1], window=1, max_width=None)
        words = "\n".join(d[4] for d in sorted(detections, key=lambda d: (d[1], d[0])))
        ui.set_text(
            f"Captured. {len(detections)} text region(s) detected. "
            f"Press 's' to re-capture." + (f"\n\n{words}" if words else ""),
            window=1,
        )
        ui.set_sections(_info_sections())
        print(f"Captured frame: {len(detections)} text region(s) detected.", flush=True)

        if args.output_image:
            cv2.imwrite(str(args.output_image), annotated[..., ::-1])

        if args.output_text:
            with open(args.output_text, "w") as f:
                for _x0, _y0, _x1, _y1, text, _confidence in detections:
                    f.write(f"{text}\n")

    _latest_frame: np.ndarray | None = None
    ui.set_windows(1, 2, titles=["Live", "Detected"])
    ui.set_title(info.name)
    ui.set_text("Press 's' to capture and run OCR on the current frame.", window=1)
    ui.on_keypress("s", lambda: do_capture(_latest_frame))
    ui.on_keypress("S", lambda: do_capture(_latest_frame))

    frames_until_capture = args.auto_capture_after_frames
    last_sections_update = 0.0
    try:
        ui.start_thread()
        while True:
            rgb_frame = outq.get(timeout=5)
            _latest_frame = rgb_frame
            ui.set_frame(rgb_frame[..., ::-1], window=0)
            now = time.monotonic()
            if now - last_sections_update >= 0.5:
                ui.set_sections(_info_sections())
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
    parser = argparse.ArgumentParser(description="On-device EasyOCR app")
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
        help=(
            "GStreamer video source, e.g. "
            '"v4l2src device=/dev/video2", '
            '"qtiqmmfsrc name=camsrc camera=0", '
            "or for a still image "
            '"filesrc location=doc.png ! decodebin ! imagefreeze"'
        ),
    )
    parser.add_argument(
        "--video-source-width",
        type=int,
        default=1024,
        help="Video width (input), default 1024",
    )
    parser.add_argument(
        "--video-source-height",
        type=int,
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
        "--output-image",
        type=Path,
        default=None,
        help=(
            "If set, save the annotated frame from each capture (pressing "
            "'s' in the browser) to this path (overwriting). Used by test.sh "
            "to verify output against a reference image."
        ),
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=None,
        help=(
            "If set, save the detected text from each capture (pressing 's' "
            "in the browser) to this path (overwriting), one line per "
            "detection. Used by test.sh to verify recognized text against a "
            "reference."
        ),
    )
    parser.add_argument(
        "--auto-capture-after-frames",
        type=int,
        default=None,
        help=(
            "If set, automatically trigger a capture (as if 's' was pressed) "
            "after this many frames have been streamed, without needing a "
            "browser. Used by test.sh to exercise the capture flow headlessly."
        ),
    )
    parser.add_argument(
        "--recognizer-workers",
        type=int,
        default=1,
        help=(
            "Number of recognizer interpreters, i.e. how many boxes are "
            "recognized in parallel per capture (each loads its own copy of the "
            "recognizer model). 1 recognizes boxes one at a time."
        ),
    )
    parser.add_argument(
        "--max-recognitions",
        type=int,
        default=20,
        help=(
            "Run the recognizer on at most this many of the detector's boxes "
            "per capture (the recognizer runs once per box and dominates "
            "capture latency on a text-dense frame). Pass 0 or a negative "
            "value to disable the cap and recognize every detected box."
        ),
    )

    args = parser.parse_args()
    if args.max_recognitions is not None and args.max_recognitions <= 0:
        args.max_recognitions = None
    main(args)
