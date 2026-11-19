#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Configurable custom object-detection postprocess example (YOLOv8).

The input source and output sink are selectable at runtime:

  * input  : H.264/MP4 file (filesrc) OR USB camera (v4l2src)
  * output : MP4 file (filesink) OR RTSP stream (qtirtspbin), also shown in
             the browser web UI

Detection decode (tensor read, NMS, label/color mapping) is done in a custom
Python postprocess callback instead of the built-in qtimlpostprocess module.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, NoReturn

import gi
import numpy as np

gi.require_version("GLib", "2.0")
gi.require_version("Gst", "1.0")
gi.require_version("GstQtiML", "1.0")

from gi.repository import GstQtiML  # noqa: E402
from qai_hub_apps_utils.bbox_processing import (  # noqa: E402
    batched_nms,
    filter_boxes,
    map_boxes_to_region,
)
from qai_hub_apps_utils.imsdk import (  # noqa: E402
    add_file_sink,
    add_file_source,
    add_rtsp_sink,
    add_usb_camera_source,
    execute_pipeline,
    get_input_region,
    get_output_tensors,
    tflite_htp_inference,
)
from qai_hub_apps_utils.input_devices import get_default_video_device  # noqa: E402
from qai_hub_apps_utils.webui import WebUI  # noqa: E402
from qimsdk import (  # noqa: E402
    Element,
    ImsdkGstLogMode,
    ImsdkLogLevel,
    MLPostprocess,
    ObjectDetections,
    Pipeline,
    SetImsdkGstLogMode,
    SetImsdkLogLevel,
    TextFilter,
    VideoFilter,
)


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """Argument parser that prints full help text on any parse error."""

    def error(self, message: str) -> NoReturn:
        self.print_help(sys.stderr)
        sys.stderr.write(f"\n{self.prog}: error: {message}\n")
        sys.exit(2)


SCORE_THRESHOLD_DEFAULT = 0.70
IOU_THRESHOLD_DEFAULT = 0.5

DEFAULT_LABEL_NAME = "unknown"
DEFAULT_LABEL_COLOR = 0x00FF00FF  # RGBA green

# The quantized TFLite export emits (bboxes, scores, class ids).
EXPECTED_TENSORS = 3

LABELS_FILE = "labels.txt"  # one class name per line, line index == class id
MODEL_FILE = "detector.tflite"


@dataclass
class Labels:
    """Per-class display names and RGBA overlay colors, indexed by class id."""

    names: np.ndarray
    colors: np.ndarray


def load_labels(path: str) -> Labels:
    # A .txt file carries no colors, so every class gets the default overlay color.
    with open(path, encoding="utf-8") as f:
        names = [line.strip() for line in f]
    return Labels(
        names=np.array(names, dtype=object),
        colors=np.full(len(names), DEFAULT_LABEL_COLOR, dtype=np.uint32),
    )


def append_detections(
    detections: ObjectDetections,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
    labels: Labels,
) -> None:
    # Class ids with no label get the default name and color.
    class_ids = class_indices.astype(int)
    names = np.full(class_ids.shape, DEFAULT_LABEL_NAME, dtype=object)
    colors = np.full(class_ids.shape, DEFAULT_LABEL_COLOR, dtype=np.uint32)
    known = (class_ids >= 0) & (class_ids < len(labels.names))
    names[known] = labels.names[class_ids[known]]
    colors[known] = labels.colors[class_ids[known]]

    for (left, top, right, bottom), score, name, color in zip(
        boxes, scores, names, colors, strict=True
    ):
        detection = GstQtiML.Detection()
        detection.left = float(left)
        detection.top = float(top)
        detection.right = float(right)
        detection.bottom = float(bottom)
        detection.confidence = float(score) * 100.0
        detection.name = str(name)
        detection.color = np.uint32(color)
        detections.append(detection)


@dataclass
class DetectionStats:
    frames: int = 0
    frames_with_objects: int = 0
    objects: int = 0


def make_detection_callback(labels: Labels, stats: DetectionStats) -> Any:
    """Return the postprocess handler decoding YOLOv8 outputs with `labels`."""

    def detection_callback(
        mlframe: Any, mlparams: Any, detections: ObjectDetections
    ) -> bool:
        if mlframe.info.n_tensors != EXPECTED_TENSORS:
            raise KeyError(
                f"Expected {EXPECTED_TENSORS} tensors, got {mlframe.info.n_tensors}"
            )

        region, resolution = get_input_region(mlparams)
        bboxes, scores, classes = get_output_tensors(mlframe)
        if not (bboxes.shape[0] == scores.shape[0] == classes.shape[0]):
            raise KeyError("Size of all three tensors must be equal!")

        bboxes = map_boxes_to_region(bboxes, resolution, region)
        bboxes, scores, classes = filter_boxes(
            bboxes, scores, classes, score_threshold=SCORE_THRESHOLD_DEFAULT
        )
        kept_boxes, kept_scores, kept_classes = batched_nms(
            IOU_THRESHOLD_DEFAULT, None, bboxes[None], scores[None], classes[None]
        )
        append_detections(
            detections, kept_boxes[0], kept_scores[0], kept_classes[0], labels
        )
        found = len(kept_boxes[0])
        stats.frames += 1
        stats.objects += found
        stats.frames_with_objects += found > 0
        return True

    return detection_callback


#  Pipeline (discrete custom-postprocess topology):
#
#    <source> -> split (tee)
#      split -> [writable copy] -> mlmuxer
#      split -> q1 -> preprocessing -> q2 -> inference -> q3
#             -> postprocessing (Python callback) -> [TextFilter] -> mlmuxer
#      mlmuxer -> q4 -> overlay -> <output>
#
#  <source> : MP4 file (hardware decode) OR USB camera, as NV12
#  <output> : MP4 file OR RTSP stream


def create_and_execute_pipeline(
    model_base_path: str,
    input_config: str,
    input_type: str,
    output: str,
    output_file: str,
    rtsp_port: int = 8900,
    rtsp_mount: str = "/live",
) -> None:
    labels = load_labels(f"{model_base_path}/{LABELS_FILE}")
    stats = DetectionStats()

    pipeline = Pipeline("ml-external-detection")

    for element in (
        Element("tee", "split"),
        Element("queue", "q1"),
        Element("qtimlvconverter", "preprocessing"),
        Element("queue", "q2"),
        tflite_htp_inference(f"{model_base_path}/{MODEL_FILE}"),
        Element("queue", "q3"),
        MLPostprocess("postprocessing").set_handler(
            make_detection_callback(labels, stats)
        ),
        Element("qtimetamux", "mlmuxer"),
        Element("queue", "q4"),
        Element("qtivoverlay", "overlay"),
    ):
        pipeline.add(element)
    # Carries the detection metadata into the muxer.
    pipeline.add_stream_filter("mlf", TextFilter())

    if input_type == "file":
        add_file_source(pipeline, input_config, tail="split")
    else:
        add_usb_camera_source(pipeline, input_config, tail="split")

    # Passthrough (video) branch into the metadata muxer. The encoder can hold
    # buffers under load, so metadata goes on a private writable copy.
    pipeline.add(Element("queue", "q_pass"))
    pipeline.add(Element("qtivtransform", "pass_transform"))
    pipeline.add_stream_filter("pass_vf", VideoFilter().format("NV12"))
    pipeline.link("split", "q_pass", "pass_transform", "pass_vf", "mlmuxer")

    # AI branch into the metadata muxer.
    pipeline.link(
        "split",
        "q1",
        "preprocessing",
        "q2",
        "inference",
        "q3",
        "postprocessing",
        "mlf",
        "mlmuxer",
    )

    pipeline.link("mlmuxer", "q4", "overlay")
    if output == "file":
        add_file_sink(pipeline, head="overlay", path=output_file)
    else:
        add_rtsp_sink(pipeline, head="overlay", port=rtsp_port, mount=rtsp_mount)

    execute_pipeline(pipeline)

    # test.sh parses these lines.
    print(f"Number of frames: {stats.frames}")
    print(f"Frames with objects: {stats.frames_with_objects}")
    print(f"Number of objects: {stats.objects}")


def main() -> None:
    if "HOME" not in os.environ:
        raise OSError("Error: HOME environment variable is not set.")

    default_base = f"{os.curdir}/models"
    default_media = f"{os.curdir}/ai_demo_sample.mp4"
    default_output = f"{os.curdir}/output.mp4"

    parser = HelpOnErrorArgumentParser(
        description="QIMSDK configurable external-postprocess YOLOv8 detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model-base-path",
        default=default_base,
        help=f"Directory holding {MODEL_FILE} and {LABELS_FILE}",
    )
    parser.add_argument(
        "-i",
        "--input-config",
        default=None,
        help="Input: file path (file) or device node like /dev/video2 (usb-camera). "
        f"Defaults to {default_media} for file, or the host's first camera.",
    )
    parser.add_argument(
        "--input-type",
        choices=["file", "usb-camera"],
        default="file",
        help="Input source type",
    )
    parser.add_argument(
        "--output",
        choices=["file", "rtsp"],
        default="rtsp",
        help="Output sink (rtsp is also shown in the browser web UI on port 8080)",
    )
    parser.add_argument(
        "--output-file",
        default=default_output,
        help="Output MP4 path when --output file",
    )
    parser.add_argument(
        "--rtsp-port", type=int, default=8900, help="RTSP server port (--output rtsp)"
    )
    parser.add_argument(
        "--rtsp-mount", default="/live", help="RTSP mount point (--output rtsp)"
    )
    args = parser.parse_args()

    input_config = args.input_config
    if input_config is None:
        input_config = (
            get_default_video_device()
            if args.input_type == "usb-camera"
            else default_media
        )
    print(f"[info] input ({args.input_type}): {input_config}")

    if args.output == "rtsp":
        ui = WebUI()
        ui.set_title("Object Detection in IMSDK")
        ui.set_rtsp_stream(f"rtsp://127.0.0.1:{args.rtsp_port}{args.rtsp_mount}")
        ui.set_text(
            f"RTSP: rtsp://<device-ip>:{args.rtsp_port}{args.rtsp_mount}  |  "
            f"input: {input_config}"
        )
        ui.start_thread()

    SetImsdkGstLogMode(ImsdkGstLogMode.ImsdkLog)
    SetImsdkLogLevel(ImsdkLogLevel.Debug)

    create_and_execute_pipeline(
        model_base_path=args.model_base_path,
        input_config=input_config,
        input_type=args.input_type,
        output=args.output,
        output_file=args.output_file,
        rtsp_port=args.rtsp_port,
        rtsp_mount=args.rtsp_mount,
    )


if __name__ == "__main__":
    main()
