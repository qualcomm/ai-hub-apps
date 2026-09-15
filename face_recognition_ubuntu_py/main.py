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
import numpy as np
import qai_hub_apps_utils.webui as ui
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.draw import draw_box_from_xyxy
from qai_hub_apps_utils.fps import FpsCounter
from qai_hub_apps_utils.input_devices import get_default_video_device
from qai_hub_apps_utils.platform import get_current_device
from qai_hub_apps_utils.quantization import dequantize, quantize
from utils.detection import crop_face, detect_faces, load_cascade
from utils.gallery import build_gallery
from utils.input_processing import (
    get_gstreamer_input_pipeline,
    load_image_rgb,
    preprocess_image,
)
from utils.model_io_processing import identify, l2_normalize

MODEL_PATH = "models/cavaface.tflite"


def build_interpreter(
    qairt_path: Path | None,
    hexagon_version: str | None,
) -> Interpreter:
    """Create a TFLite interpreter, preferring the QNN NPU delegate.

    If ``qairt_path`` is given and the QNN TFLite delegate loads successfully,
    inference runs on the Hexagon NPU. Otherwise (no path, missing library, or
    a delegate that fails to initialize) the interpreter falls back to plain
    CPU execution, which needs no QAIRT SDK and produces identical results for
    this float model.

    Parameters
    ----------
    qairt_path
        Path to the QAIRT SDK root, or None to run on CPU.
    hexagon_version
        Hexagon version of the device (e.g. "v73"), used to locate skel libs.
        May be None when running on CPU (``qairt_path`` is None).

    Returns
    -------
    Interpreter
        An allocated TFLite interpreter ready for inference.
    """
    if qairt_path is not None:
        delegate_path = (
            qairt_path / "lib" / "aarch64-oe-linux-gcc11.2" / "libQnnTFLiteDelegate.so"
        )
        if delegate_path.exists():
            try:
                delegate = Delegate(
                    str(delegate_path),
                    {
                        "backend_type": "htp",
                        "htp_performance_mode": "2",
                        "library_path": str(
                            qairt_path
                            / "lib"
                            / "aarch64-oe-linux-gcc11.2"
                            / "libQnnHtp.so"
                        ),
                        "skel_library_dir": str(
                            qairt_path
                            / "lib"
                            / f"hexagon-{hexagon_version}"
                            / "unsigned"
                        ),
                    },
                )
                interpreter = Interpreter(MODEL_PATH, experimental_delegates=[delegate])
                interpreter.allocate_tensors()
                print("Backend: QNN NPU delegate (Hexagon)", flush=True)
                return interpreter
            except Exception as exc:
                print(
                    f"NPU delegate failed to initialize ({exc}); using CPU",
                    flush=True,
                )
        else:
            print(
                f"QNN delegate not found at {delegate_path}; using CPU",
                flush=True,
            )
    else:
        print("No --qairt-path provided; using CPU", flush=True)

    interpreter = Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    print("Backend: CPU", flush=True)
    return interpreter


def _set_input(
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    rgb_input: np.ndarray,
) -> None:
    """Quantize (if needed) and feed the preprocessed RGB input into the model.

    Parameters
    ----------
    interpreter
        TFLite interpreter for CavaFace.
    input_details
        Input tensor details from interpreter.get_input_details().
    rgb_input
        Preprocessed RGB image of shape [1, H, W, 3], dtype float32 in [0, 1].
    """
    detail = input_details[0]
    if np.issubdtype(detail["dtype"], np.integer):
        input_val = quantize(
            rgb_input,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        input_val = rgb_input.astype(detail["dtype"])
    interpreter.set_tensor(detail["index"], input_val)


def _get_output(
    interpreter: Interpreter,
    detail: dict[str, Any],
) -> np.ndarray:
    """Read one output tensor, dequantizing it if the model is quantized.

    The model emits a [1, EMBEDDING_DIM] tensor; the leading batch dimension is
    dropped to yield a 1-D embedding.

    Parameters
    ----------
    interpreter
        TFLite interpreter for CavaFace.
    detail
        A single entry from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Face embedding of shape [EMBEDDING_DIM], dtype float32.
    """
    tensor = interpreter.get_tensor(detail["index"])
    if np.issubdtype(detail["dtype"], np.integer):
        tensor = dequantize(
            tensor,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    embedding = tensor.reshape(-1)
    if embedding.size != C.EMBEDDING_DIM:
        raise ValueError(
            f"Expected a {C.EMBEDDING_DIM}-dim embedding, got {embedding.size}"
        )
    return embedding


def embed_crop(
    rgb_crop: np.ndarray,
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    output_details: list[dict[str, Any]],
) -> np.ndarray:
    """Run CavaFace on a cropped face region and return its embedding.

    Parameters
    ----------
    rgb_crop
        RGB face crop of shape [H, W, 3], dtype uint8.
    interpreter
        TFLite interpreter for CavaFace.
    input_details
        Input tensor details from interpreter.get_input_details().
    output_details
        Output tensor details from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Unit-length face embedding of shape [EMBEDDING_DIM], dtype float32.
    """
    model_input = preprocess_image(rgb_crop)
    _set_input(interpreter, input_details, model_input)
    interpreter.invoke()
    # CavaFace already emits L2-normalized embeddings; normalize defensively so
    # cosine similarity stays a plain dot product for any model variant.
    return l2_normalize(_get_output(interpreter, output_details[0]))


def recognize_faces(
    rgb_frame: np.ndarray,
    embed_fn: Any,
    cascade: cv2.CascadeClassifier,
    gallery: dict[str, np.ndarray],
    threshold: float,
    *,
    whole_image_fallback: bool,
) -> list[tuple[str, float]]:
    """Detect, identify, and annotate every face in an RGB frame in place.

    Parameters
    ----------
    rgb_frame
        RGB image of shape [H, W, 3], dtype uint8. Boxes and labels are drawn
        onto this array in place.
    embed_fn
        Callable mapping an RGB face crop to its embedding of shape [D].
    cascade
        Loaded Haar cascade used to locate faces.
    gallery
        Mapping from identity name to prototype embedding.
    threshold
        Minimum cosine similarity for a match; below it a face is "Unknown".
    whole_image_fallback
        If True and no face is detected, treat the whole frame as one face. Used
        for single images (which may be tight pre-cropped faces); left False for
        live frames so an empty scene is not labeled.

    Returns
    -------
    list[tuple[str, float]]
        The (name, score) result for each annotated face, in detection order.
    """
    boxes = detect_faces(rgb_frame, cascade)
    if not boxes and whole_image_fallback:
        height, width = rgb_frame.shape[:2]
        boxes = [(0, 0, width, height)]

    results: list[tuple[str, float]] = []
    for box in boxes:
        crop = crop_face(rgb_frame, box, C.BOX_MARGIN)
        if crop.size == 0:
            continue
        name, score = identify(embed_fn(crop), gallery, threshold)
        color = C.KNOWN_COLOR if name != C.UNKNOWN_LABEL else C.UNKNOWN_COLOR
        draw_box_from_xyxy(
            rgb_frame,
            (box[0], box[1]),
            (box[2], box[3]),
            color=color,
            size=C.BOX_THICKNESS,
            text=f"{name} {score:.2f}",
        )
        results.append((name, score))
    return results


def run_image(
    args: argparse.Namespace,
    embed_fn: Any,
    cascade: cv2.CascadeClassifier,
    gallery: dict[str, np.ndarray],
) -> None:
    """Recognize faces in a single image, printing and optionally saving results."""
    rgb_frame = load_image_rgb(args.image)
    results = recognize_faces(
        rgb_frame,
        embed_fn,
        cascade,
        gallery,
        args.recognition_threshold,
        whole_image_fallback=True,
    )

    if not results:
        print("No faces detected.", flush=True)
    for name, score in results:
        print(f"{name}: {score:.3f}", flush=True)

    if args.output:
        cv2.imwrite(args.output, rgb_frame[..., ::-1])
        print(f"Wrote annotated image to {args.output}", flush=True)


def run_live(
    args: argparse.Namespace,
    embed_fn: Any,
    cascade: cv2.CascadeClassifier,
    gallery: dict[str, np.ndarray],
) -> None:
    """Recognize faces on a live camera stream, serving annotated frames over HTTP."""
    # GStreamer (gi) is imported lazily so --image / --list-devices work without
    # PyGObject or the system GStreamer stack installed.
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    outq: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)

    def on_new_sample(sink: Any) -> Any:
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")

        # Map buffer memory as read-only
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            rowstride = mapinfo.size // height
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8, count=height * rowstride)
            arr = arr.reshape(height, rowstride)[:, : width * 3].copy()
            arr = arr.reshape((height, width, 3))
        finally:
            buf.unmap(mapinfo)

        with contextlib.suppress(queue.Full):
            outq.put_nowait(arr)

        return Gst.FlowReturn.OK

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

            recognize_faces(
                rgb_frame,
                embed_fn,
                cascade,
                gallery,
                args.recognition_threshold,
                whole_image_fallback=False,
            )

            fps_counter.tick()

            ui.set_frame(rgb_frame[..., ::-1])

    except queue.Empty:
        print("Timed out waiting for input! Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)


def main(args: argparse.Namespace) -> None:
    """Load the model and gallery, then dispatch to the requested mode."""
    if args.list_devices:
        subprocess.call(["v4l2-ctl", "--list-devices"])
        return

    if args.qairt_path is not None and not args.hexagon_version:
        raise SystemExit(
            "Unknown Hexagon version for this device. "
            "Pass it with --hexagon-version <e.g. v73>."
        )

    interpreter = build_interpreter(args.qairt_path, args.hexagon_version)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def embed_fn(rgb_crop: np.ndarray) -> np.ndarray:
        return embed_crop(rgb_crop, interpreter, input_details, output_details)

    cascade = load_cascade()
    gallery = build_gallery(args.gallery_dir, embed_fn, cascade=cascade)

    if args.image:
        run_image(args, embed_fn, cascade, gallery)
    else:
        run_live(args, embed_fn, cascade, gallery)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CavaFace Face Recognition")
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
    group.add_argument(
        "--image",
        type=str,
        help="Recognize faces in a single image instead of a live camera",
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default=None,
        help="Directory of enrolled identities (see README for layout). "
        "Required unless --list-devices.",
    )
    parser.add_argument(
        "--recognition-threshold",
        type=float,
        default=C.RECOGNITION_THRESHOLD,
        help="Minimum cosine similarity for a match, in [0, 1] "
        f"(default {C.RECOGNITION_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the annotated image (only used with --image)",
    )
    parser.add_argument(
        "--video-source-width",
        type=int,
        required=False,
        default=C.VIDEO_WIDTH,
        help=f"Video width (input), default {C.VIDEO_WIDTH}",
    )
    parser.add_argument(
        "--video-source-height",
        type=int,
        required=False,
        default=C.VIDEO_HEIGHT,
        help=f"Video height (input), default {C.VIDEO_HEIGHT}",
    )
    parser.add_argument(
        "--qairt-path",
        type=Path,
        default=None,
        help="Path to QAIRT SDK root. If omitted, inference runs on CPU.",
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
    if not args.list_devices and not args.gallery_dir:
        parser.error("--gallery-dir is required unless --list-devices is given")
    main(args)
