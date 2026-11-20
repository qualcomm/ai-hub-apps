#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""TFLite/NPU image classification with native IMSDK MobileNet postprocessing."""

import argparse
import json
import os
import signal
import threading
from types import FrameType
from typing import Any

from qai_hub_apps_utils.imsdk import (
    add_file_sink,
    add_file_source,
    add_rtsp_sink,
    add_usb_camera_source,
    tflite_htp_inference,
)
from qai_hub_apps_utils.input_devices import get_default_video_device
from qai_hub_apps_utils.webui import WebUI
from qimsdk import (
    Element,
    ImsdkGstLogMode,
    ImsdkLogLevel,
    Pipeline,
    SetImsdkGstLogMode,
    SetImsdkLogLevel,
    TextFilter,
    VideoFilter,
)

MODEL_FILE = "classifier.tflite"
LABELS_FILE = "labels.txt"
POSTPROCESS_MODULE = "mobilenet"


def log_classification_results(pad: Any, info: Any, errors: list[Exception]) -> Any:
    """Observe classification metadata."""
    from gi.repository import Gst

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    ok, mapped = buffer.map(Gst.MapFlags.READ)
    if not ok:
        errors.append(RuntimeError("Could not read classification metadata"))
        return Gst.PadProbeReturn.OK
    try:
        text = bytes(mapped.data).decode("utf-8").rstrip(chr(0))
        structure = Gst.Structure.new_from_string(
            f"classification-buffer, entries={text};"
        )
        for entry in structure.get_value("entries"):
            if entry.get_name() == "ImageClassification":
                for label in entry.get_value("labels"):
                    prediction = {
                        "label": label.get_name(),
                        "confidence": label.get_value("confidence"),
                    }
                    print("[classification] " + json.dumps(prediction), flush=True)
    except Exception as error:
        errors.append(error)
    finally:
        buffer.unmap(mapped)
    return Gst.PadProbeReturn.OK


def execute_pipeline(pipeline: Pipeline) -> None:
    """Propagate worker failures and drain file output on Ctrl+C."""
    interrupted = threading.Event()
    errors: list[BaseException] = []

    def on_sigint(signum: int, frame: FrameType | None) -> None:
        interrupted.set()

    def run() -> None:
        try:
            pipeline.execute()
        except BaseException as error:
            errors.append(error)

    previous_handler = signal.signal(signal.SIGINT, on_sigint)
    worker = threading.Thread(target=run, daemon=False)
    try:
        worker.start()
        while worker.is_alive() and not interrupted.is_set():
            worker.join(timeout=0.2)
        if interrupted.is_set() and worker.is_alive():
            pipeline.stop()
            # Do not abandon the muxer while it writes the MP4 trailer.
            worker.join()
        if errors:
            raise errors[0]
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def create_and_execute_pipeline(
    model_base_path: str,
    input_config: str,
    input_type: str,
    output: str,
    output_file: str,
    rtsp_port: int = 8900,
    rtsp_mount: str = "/live",
    log_classifications: bool = False,
) -> None:
    """Build classification metadata overlay and hardware-encoded output."""
    pipeline = Pipeline("ml-image-classification")
    for element in (
        Element("tee", "split"),
        Element("queue", "q_ai"),
        Element("qtimlvconverter", "preprocessing"),
        Element("queue", "q_infer"),
        tflite_htp_inference(f"{model_base_path}/{MODEL_FILE}"),
        Element("queue", "q_post"),
        Element("qtimlpostprocess", "postprocessing").set(
            "module", POSTPROCESS_MODULE, "labels", f"{model_base_path}/{LABELS_FILE}"
        ),
        Element("qtimetamux", "metamux"),
        Element("queue", "q_overlay"),
        Element("qtivoverlay", "overlay"),
    ):
        pipeline.add(element)
    pipeline.add_stream_filter("classification_text", TextFilter())
    metadata_errors: list[Exception] = []
    if log_classifications:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        pad = pipeline.get("postprocessing").get_raw().get_static_pad("src")
        pad.add_probe(
            Gst.PadProbeType.BUFFER, log_classification_results, metadata_errors
        )

    if input_type == "file":
        add_file_source(pipeline, input_config, tail="split")
    else:
        add_usb_camera_source(pipeline, input_config, tail="split")

    # Give the metadata attachment branch a private video buffer for encoding.
    pipeline.add(Element("queue", "q_video"))
    pipeline.add(Element("qtivtransform", "video_copy"))
    pipeline.add_stream_filter("video_nv12", VideoFilter().format("NV12"))
    pipeline.link("split", "q_video", "video_copy", "video_nv12", "metamux")
    pipeline.link(
        "split",
        "q_ai",
        "preprocessing",
        "q_infer",
        "inference",
        "q_post",
        "postprocessing",
        "classification_text",
        "metamux",
    )
    pipeline.link("metamux", "q_overlay", "overlay")
    if output == "file":
        add_file_sink(pipeline, head="overlay", path=output_file)
    else:
        add_rtsp_sink(pipeline, head="overlay", port=rtsp_port, mount=rtsp_mount)
    execute_pipeline(pipeline)
    if metadata_errors:
        raise RuntimeError(
            "Failed to log classification metadata"
        ) from metadata_errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image classification with native IMSDK MobileNet postprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model-base-path",
        default="./models",
        help=f"Directory holding {MODEL_FILE} and {LABELS_FILE}",
    )
    parser.add_argument(
        "-i",
        "--input-config",
        default=None,
        help="H.264 MP4 path or USB V4L2 device node",
    )
    parser.add_argument("--input-type", choices=["file", "usb-camera"], default="file")
    parser.add_argument("--output", choices=["file", "rtsp"], default="file")
    parser.add_argument("--output-file", default="./output.mp4")
    parser.add_argument("--rtsp-port", type=int, default=8900)
    parser.add_argument("--rtsp-mount", default="/live")
    parser.add_argument(
        "--log-level",
        choices=["error", "warning", "info", "debug"],
        default="info",
        help="IMSDK logging verbosity",
    )
    parser.add_argument(
        "--log-classifications",
        action="store_true",
        help="Log native classification labels",
    )
    args = parser.parse_args()
    if not 1 <= args.rtsp_port <= 65535:
        parser.error("--rtsp-port must be between 1 and 65535")
    if not args.rtsp_mount.startswith("/"):
        parser.error("--rtsp-mount must start with '/'")

    input_config = args.input_config
    if input_config is None:
        input_config = (
            get_default_video_device()
            if args.input_type == "usb-camera"
            else "./classification_sample.mp4"
        )
    model_base_path = os.path.expanduser(args.model_base_path)
    input_config = os.path.expanduser(input_config)
    output_file = os.path.expanduser(args.output_file)
    for path in (f"{model_base_path}/{MODEL_FILE}", f"{model_base_path}/{LABELS_FILE}"):
        if not os.path.isfile(path):
            parser.error(f"Required asset not found: {path}")
    if args.input_type == "file" and not os.path.isfile(input_config):
        parser.error(f"Input video not found: {input_config}")

    if args.output == "rtsp":
        ui = WebUI()
        ui.set_title("Image Classification in IMSDK")
        ui.set_rtsp_stream(f"rtsp://127.0.0.1:{args.rtsp_port}{args.rtsp_mount}")
        ui.set_text(f"Input: {input_config} | Postprocess: {POSTPROCESS_MODULE}")
        ui.start_thread()

    SetImsdkGstLogMode(ImsdkGstLogMode.ImsdkLog)
    log_levels = {
        "error": ImsdkLogLevel.Error,
        "warning": ImsdkLogLevel.Warning,
        "info": ImsdkLogLevel.Info,
        "debug": ImsdkLogLevel.Debug,
    }
    SetImsdkLogLevel(log_levels[args.log_level])
    create_and_execute_pipeline(
        model_base_path=model_base_path,
        input_config=input_config,
        input_type=args.input_type,
        output=args.output,
        output_file=output_file,
        rtsp_port=args.rtsp_port,
        rtsp_mount=args.rtsp_mount,
        log_classifications=args.log_classifications,
    )


if __name__ == "__main__":
    main()
