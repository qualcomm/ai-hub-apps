# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import contextlib
import queue
import subprocess
import warnings
from pathlib import Path
from typing import Any

import cv2
import gi
import numpy as np
import qai_hub_apps_utils.webui as ui
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.fps import FpsCounter
from qai_hub_apps_utils.quantization import dequantize, quantize
from utils.input_processing import get_gstreamer_input_pipeline
from utils.model_io_processing import (
    blend_mask,
    blur_background,
    build_soft_alpha,
    composite_background,
    decode_mask,
    postprocess_mask,
    resize_to_fit,
)
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
        TFLite interpreter for SINet.
    input_details
        Input tensor details from interpreter.get_input_details().
    rgb_input
        Preprocessed RGB image of shape [1, H, W, 3], dtype uint8 in range [0, 255].
    """
    detail = input_details[0]
    if np.issubdtype(detail["dtype"], np.integer):
        # Quantized model: the input range is normalized [0, 1].
        normalized = rgb_input.astype(np.float32) / 255.0
        input_val = quantize(
            normalized,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        # Float model expects RGB in [0, 1].
        input_val = (rgb_input.astype(np.float32) / 255.0).astype(detail["dtype"])
    if len(detail["shape"]) == 4 and detail["shape"][1] == 3:
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
        TFLite interpreter for SINet.
    detail
        A single entry from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Output tensor as float, shape [1, H, W, 2] (NHWC).
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
) -> np.ndarray:
    """Run the segmentation model on a single RGB frame and decode the mask.

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
        Model input (height, width) to resize the frame to.

    Returns
    -------
    np.ndarray
        Foreground probability map at the model's output resolution, dtype
        float32, in [0, 1].
    """
    input_height, input_width = input_size
    resized = cv2.resize(
        rgb_frame, (input_width, input_height), interpolation=cv2.INTER_LINEAR
    )
    input_val = np.expand_dims(resized, axis=0)

    _set_input(interpreter, input_details, input_val)
    interpreter.invoke()

    model_output = _get_output(interpreter, output_details[0])
    return decode_mask(model_output)


def main(args: argparse.Namespace) -> None:
    if args.list_devices:
        subprocess.call(["v4l2-ctl", "--list-devices"])
        return

    bg_image: np.ndarray | None = None
    if args.background in ("overlay", "blur"):
        bg_mode = args.background
    else:
        bg_path = Path(args.background)
        bgr = cv2.imread(str(bg_path), cv2.IMREAD_COLOR) if bg_path.exists() else None
        if bgr is None:
            raise FileNotFoundError(
                f"Background image not found or unreadable: {bg_path}"
            )
        bg_image = resize_to_fit(
            bgr[..., ::-1], (args.video_source_width, args.video_source_height)
        )
        bg_mode = "image"

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
    try:
        ui.start_thread()
        while True:
            rgb_frame = outq.get(timeout=5)

            prob = run_inference(
                rgb_frame,
                interpreter,
                input_details,
                output_details,
                input_size,
            )

            frame_h, frame_w = rgb_frame.shape[:2]
            if bg_mode == "overlay":
                mask = postprocess_mask(prob, (frame_w, frame_h))
                blend_mask(rgb_frame, mask)
            else:
                alpha = build_soft_alpha(prob, (frame_w, frame_h))
                if bg_mode == "blur":
                    background = blur_background(rgb_frame)
                else:
                    assert bg_image is not None
                    background = bg_image
                composite_background(rgb_frame, alpha, background)

            fps_counter.tick()

            ui.set_frame(rgb_frame[..., ::-1])

    except queue.Empty:
        print("Timed out waiting for input! Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SINet Portrait Segmentation")
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
        "--background",
        type=str,
        default="blur",
        help=(
            'Background mode: "blur" (default) blurs the background, "overlay" '
            "tints the person with a solid color, or a path to an image to place "
            "behind the person"
        ),
    )

    args = parser.parse_args()
    main(args)
