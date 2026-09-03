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
from qai_hub_apps_utils.bbox_processing import batched_nms
from qai_hub_apps_utils.fps import FpsCounter
from qai_hub_apps_utils.input_devices import get_default_video_device
from qai_hub_apps_utils.platform import get_current_device
from qai_hub_apps_utils.quantization import dequantize, quantize
from utils.draw import draw_3d_box
from utils.geometry import build_projection_matrix
from utils.input_processing import get_gstreamer_input_pipeline
from utils.model_io_processing import (
    decode_3d_box,
    load_labels,
    preprocess_crop,
    preprocess_frame,
    select_detections,
)
from utils.model_metadata import ModelIO, find_model, load_model_metadata

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
    interpreter: Interpreter, detail: dict[str, Any], normalized_input: np.ndarray
) -> None:
    """Quantize (if needed) and feed a normalized input into the model.

    Parameters
    ----------
    interpreter
        TFLite interpreter to feed.
    detail
        The interpreter's input tensor detail.
    normalized_input
        Batched float32 input in [0, 1].
    """
    if np.issubdtype(detail["dtype"], np.integer):
        input_val = quantize(
            normalized_input,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        input_val = normalized_input.astype(detail["dtype"])
    interpreter.set_tensor(detail["index"], input_val)


def _get_outputs(
    interpreter: Interpreter, output_names: list[str]
) -> tuple[np.ndarray, ...]:
    """Read the named output tensors, dequantizing any that are quantized.

    Parameters
    ----------
    interpreter
        TFLite interpreter to read from.
    output_names
        Names of the output tensors to read, in the wanted order.

    Returns
    -------
    tuple[np.ndarray, ...]
        One dequantized tensor per requested name.

    Raises
    ------
    KeyError
        If the interpreter does not expose one of the requested outputs.
    """
    details = {detail["name"]: detail for detail in interpreter.get_output_details()}
    outputs = []
    for name in output_names:
        detail = details[name]
        tensor = interpreter.get_tensor(detail["index"])
        if (
            np.issubdtype(detail["dtype"], np.integer)
            and detail["quantization_parameters"]["scales"].size
        ):
            tensor = dequantize(
                tensor,
                zero_points=detail["quantization_parameters"]["zero_points"],
                scales=detail["quantization_parameters"]["scales"],
            )
        outputs.append(tensor)
    return tuple(outputs)


def run_inference(
    rgb_frame: np.ndarray,
    detector: Interpreter,
    detector_io: ModelIO,
    box3d: Interpreter,
    box3d_io: ModelIO,
    labels: list[str],
    proj_matrix: np.ndarray,
) -> list[tuple[float, np.ndarray, np.ndarray, str]]:
    """Detect objects in a frame and lift each detection to a 3D box.

    Parameters
    ----------
    rgb_frame
        RGB image of shape [H, W, 3] and dtype uint8.
    detector
        TFLite interpreter for the 2D detector.
    detector_io
        I/O contract of the 2D detector.
    box3d
        TFLite interpreter for the 3D box head.
    box3d_io
        I/O contract of the 3D box head.
    labels
        Class labels indexed by the detector's class index.
    proj_matrix
        3x4 camera-to-image projection matrix.

    Returns
    -------
    list[tuple[float, np.ndarray, np.ndarray, str]]
        One (orientation, dimensions, location, label) per detected object.
    """
    frame_height, frame_width = rgb_frame.shape[:2]

    detector_input = detector.get_input_details()[0]
    _set_input(
        detector,
        detector_input,
        preprocess_frame(
            rgb_frame,
            detector_io.input_height,
            detector_io.input_width,
            detector_io.channels_first,
        ),
    )
    detector.invoke()
    boxes, scores, class_indices = _get_outputs(detector, list(C.DETECTOR_OUTPUTS))

    boxes, scores, class_indices = batched_nms(
        C.NMS_IOU_THRESHOLD,
        C.NMS_SCORE_THRESHOLD,
        boxes,
        scores,
        class_indices.astype(np.int32),
    )
    detections = select_detections(
        boxes[0],
        scores[0],
        class_indices[0],
        (detector_io.input_height, detector_io.input_width),
        (frame_height, frame_width),
    )

    box3d_input = box3d.get_input_details()[0]
    results = []
    for box_2d, _, class_index in detections:
        label = labels[class_index] if class_index < len(labels) else ""
        kitti_class = C.COCO_TO_KITTI.get(label)
        if kitti_class is None:
            continue

        _set_input(
            box3d,
            box3d_input,
            preprocess_crop(
                rgb_frame,
                box_2d,
                box3d_io.input_height,
                box3d_io.input_width,
                box3d_io.channels_first,
            ),
        )
        box3d.invoke()
        orient, conf, dim = _get_outputs(box3d, list(C.BOX3D_OUTPUTS))

        decoded = decode_3d_box(
            orient[0],
            conf[0],
            dim[0],
            kitti_class,
            box_2d,
            proj_matrix,
            frame_width,
        )
        if decoded is None:
            continue
        results.append((*decoded, label))

    return results


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

    # The bundle ships two models; resolve each by its output names so the app
    # does not depend on their file names.
    models_dir = Path(C.MODELS_DIR)
    metadata = load_model_metadata(models_dir)
    detector_io = find_model(metadata, C.DETECTOR_OUTPUTS)
    box3d_io = find_model(metadata, C.BOX3D_OUTPUTS)
    labels = load_labels(str(models_dir / C.LABELS_FILE))

    detector = Interpreter(
        str(models_dir / detector_io.filename), experimental_delegates=[delegate]
    )
    box3d = Interpreter(
        str(models_dir / box3d_io.filename), experimental_delegates=[delegate]
    )
    detector.allocate_tensors()
    box3d.allocate_tensors()

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
    proj_matrix = None
    try:
        ui.start_thread()
        while True:
            rgb_frame = outq.get(timeout=5)

            if proj_matrix is None:
                frame_height, frame_width = rgb_frame.shape[:2]
                proj_matrix = build_projection_matrix(
                    frame_width, frame_height, args.hfov
                )

            for orientation, dimension, location, label in run_inference(
                rgb_frame,
                detector,
                detector_io,
                box3d,
                box3d_io,
                labels,
                proj_matrix,
            ):
                draw_3d_box(
                    rgb_frame, proj_matrix, orientation, dimension, location, label
                )

            fps_counter.tick()

            ui.set_frame(rgb_frame[..., ::-1])

    except queue.Empty:
        print("Timed out waiting for input! Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-Deep-BOX 3D Object Detection")
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
        "--hfov",
        type=float,
        default=C.DEFAULT_HFOV_DEG,
        help=(
            "Horizontal field of view of the camera in degrees, used to place "
            f"boxes in 3D, default {C.DEFAULT_HFOV_DEG}"
        ),
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
