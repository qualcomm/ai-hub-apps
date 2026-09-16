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

import gi
import numpy as np
import qai_hub_apps_utils.webui as ui
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.fps import FpsCounter
from qai_hub_apps_utils.image_processing import resize_pad
from qai_hub_apps_utils.input_devices import get_default_video_device
from qai_hub_apps_utils.platform import get_current_device
from qai_hub_apps_utils.quantization import dequantize, quantize
from utils.draw import draw_skel_and_kp
from utils.input_processing import get_gstreamer_input_pipeline
from utils.model_io_processing import decode_multiple_poses

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
        TFLite interpreter for Posenet.
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
    interpreter.set_tensor(detail["index"], input_val)


def _get_output(
    interpreter: Interpreter,
    detail: dict[str, Any],
) -> np.ndarray:
    """Read one output tensor, dequantizing it if the model is quantized.

    The model emits channels-first tensors of shape [1, C, H, W]; the leading
    batch dimension is dropped to yield (C, H, W) as expected by the decoder.

    Parameters
    ----------
    interpreter
        TFLite interpreter for Posenet.
    detail
        A single entry from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Output tensor with shape (C, H, W).
    """
    tensor = interpreter.get_tensor(detail["index"])
    if np.issubdtype(detail["dtype"], np.integer):
        tensor = dequantize(
            tensor,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    # [1, C, H, W] -> (C, H, W)
    return tensor.squeeze(0)


def run_inference(
    rgb_frame: np.ndarray,
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    output_details: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Posenet on a single RGB frame and decode pose keypoints.

    Parameters
    ----------
    rgb_frame
        Input RGB image as a numpy array of shape [H, W, 3], dtype uint8.
    interpreter
        TFLite interpreter for Posenet.
    input_details
        Input tensor details from interpreter.get_input_details().
    output_details
        Output tensor details from interpreter.get_output_details().

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        pose_scores, keypoint_scores and keypoint_coords. Coordinates are in
        (y, x) format mapped back to the original frame.
    """
    input_val, scale, pad = resize_pad(rgb_frame, (C.INPUT_HEIGHT, C.INPUT_WIDTH))
    input_val = np.expand_dims(input_val, axis=0)

    _set_input(interpreter, input_details, input_val)
    interpreter.invoke()

    # Outputs follow the model's export order:
    #   heatmaps, offsets, displacement_fwd, displacement_bwd
    heatmaps = _get_output(interpreter, output_details[0])
    offsets = _get_output(interpreter, output_details[1])
    displacement_fwd = _get_output(interpreter, output_details[2])
    displacement_bwd = _get_output(interpreter, output_details[3])

    pose_scores, keypoint_scores, keypoint_coords = decode_multiple_poses(
        heatmaps,
        offsets,
        displacement_fwd,
        displacement_bwd,
    )

    # Map (y, x) keypoint coordinates from network space back to the original frame.
    pad_left, pad_top = pad
    keypoint_coords[..., 0] = (keypoint_coords[..., 0] - pad_top) / scale
    keypoint_coords[..., 1] = (keypoint_coords[..., 1] - pad_left) / scale

    return pose_scores, keypoint_scores, keypoint_coords


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
        "models/posenet_mobilenet.tflite", experimental_delegates=[delegate]
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

            pose_scores, keypoint_scores, keypoint_coords = run_inference(
                rgb_frame,
                interpreter,
                input_details,
                output_details,
            )

            draw_skel_and_kp(
                rgb_frame,
                pose_scores,
                keypoint_scores,
                keypoint_coords,
            )

            fps_counter.tick()

            ui.set_frame(rgb_frame[..., ::-1])

    except queue.Empty:
        print("Timed out waiting for input! Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Posenet Pose Estimation")
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

    args = parser.parse_args()
    main(args)
