# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Building blocks for IMSDK (``qimsdk``) GStreamer apps.

``qimsdk`` and PyGObject ship with the IMSDK base image's system Python rather
than from pip, so this module has no requirements file.

Source helpers add their elements and link them ending at ``tail``; sink helpers
add theirs and link from ``head``. Element names are fixed, so add each kind of
source or sink at most once per pipeline.
"""

import signal
import threading
from types import FrameType
from typing import Any

import numpy as np
from qimsdk import Element, Pipeline, VideoFilter


def add_file_source(pipeline: Pipeline, path: str, tail: str) -> None:
    """Decode an H.264 MP4 file with the hardware decoder, as NV12, into `tail`.

    Parameters
    ----------
    pipeline
        Pipeline to add the source to.
    path
        Path of the MP4 file.
    tail
        Name of the already-added element the decoded stream feeds.
    """
    pipeline.add(Element("filesrc", "src").set("location", path))
    pipeline.add(Element("qtdemux", "demux"))
    pipeline.add(Element("h264parse", "parse"))
    pipeline.add(
        Element("v4l2h264dec", "decoder").set("capture-io-mode", 4, "output-io-mode", 4)
    )
    # The hardware decoder must be followed by a queue before any other stage.
    pipeline.add(Element("queue", "q_decoded"))
    pipeline.add_stream_filter("src_nv12", VideoFilter().format("NV12"))
    pipeline.link("src", "demux", "parse", "decoder", "q_decoded", "src_nv12", tail)


def add_usb_camera_source(pipeline: Pipeline, device: str, tail: str) -> None:
    """Capture YUY2 from a USB (UVC) camera and convert it to NV12, into `tail`.

    Parameters
    ----------
    pipeline
        Pipeline to add the source to.
    device
        V4L2 node of the camera, e.g. ``/dev/video2``.
    tail
        Name of the already-added element the captured stream feeds.
    """
    pipeline.add(Element("v4l2src", "src").set("device", device))
    pipeline.add_stream_filter("src_yuy2", VideoFilter().format("YUY2"))
    pipeline.add(Element("qtivtransform", "src_transform"))
    pipeline.add_stream_filter("src_nv12", VideoFilter().format("NV12"))
    pipeline.link("src", "src_yuy2", "src_transform", "src_nv12", tail)


def _add_h264_encoder(pipeline: Pipeline, config_interval: int | None = None) -> None:
    pipeline.add(
        Element("v4l2h264enc", "encoder").set("capture-io-mode", 4, "output-io-mode", 4)
    )
    parse = Element("h264parse", "parse_out")
    if config_interval is not None:
        parse.set("config-interval", config_interval)
    pipeline.add(parse)


def add_file_sink(pipeline: Pipeline, head: str, path: str) -> None:
    """Hardware-encode the NV12 stream from `head` to an H.264 MP4 at `path`.

    The pipeline is set to send EOS on stop, so the MP4 is finalized rather
    than left without its index when the run is interrupted.

    Parameters
    ----------
    pipeline
        Pipeline to add the sink to.
    head
        Name of the element whose output is recorded.
    path
        Output MP4 path.
    """
    _add_h264_encoder(pipeline)
    pipeline.add(Element("mp4mux", "mux"))
    pipeline.add(Element("filesink", "sink").set("location", path))
    pipeline.link(head, "encoder", "parse_out", "mux", "sink")
    pipeline.eos(True)


def add_rtsp_sink(
    pipeline: Pipeline,
    head: str,
    port: int = 8900,
    mount: str = "/live",
    address: str = "0.0.0.0",
) -> None:
    """Hardware-encode the NV12 stream from `head` and serve it over RTSP.

    Clients connect to ``rtsp://<address>:<port><mount>``.

    Parameters
    ----------
    pipeline
        Pipeline to add the sink to.
    head
        Name of the element whose output is served.
    port
        RTSP server port.
    mount
        RTSP mount point.
    address
        Address the RTSP server binds to.
    """
    # config-interval=1 re-sends SPS/PPS, so clients joining mid-stream can decode.
    _add_h264_encoder(pipeline, config_interval=1)
    # mode=sync paces a non-live (file) source to the client's read rate instead
    # of draining it to EOS; the server stays up waiting for clients.
    pipeline.add(
        Element("qtirtspbin", "rtsp").set(
            "address", address, "port", str(port), "mpoint", mount, "mode", "sync"
        )
    )
    pipeline.link(head, "encoder", "parse_out", "rtsp")


def tflite_htp_inference(model_path: str, name: str = "inference") -> Element:
    """Return a ``qtimltflite`` element running `model_path` on the NPU.

    Parameters
    ----------
    model_path
        Path to the ``.tflite`` model.
    name
        Element name.

    Returns
    -------
    Element
        The inference element, using the QNN HTP external delegate.
    """
    return Element("qtimltflite", name).set(
        "delegate",
        "external",
        "external-delegate-path",
        "libQnnTFLiteDelegate.so",
        "external-delegate-options",
        "QNNExternalDelegate,backend_type=htp,log_level=(string)1;",
        "model",
        model_path,
    )


def get_input_region(
    mlparams: Any,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Read the model input region and tensor size from postprocess ``mlparams``.

    Parameters
    ----------
    mlparams
        The ``mlparams`` an ``MLPostprocess`` handler receives.

    Returns
    -------
    tuple[tuple[int, int, int, int], tuple[int, int]]
        ``((x, y, width, height), (tensor_width, tensor_height))``, where the
        region is where the frame was placed in the model input, in model pixels.

    Raises
    ------
    KeyError
        If `mlparams` lacks the tensor size or the region.
    """
    ok_w, tensor_w = mlparams.get_uint("input-tensor-width")
    ok_h, tensor_h = mlparams.get_uint("input-tensor-height")
    if not (ok_h and ok_w):
        raise KeyError("Missing input tensor width/height in mlparams")

    ok_x, region_x = mlparams.get_int("input-region-x")
    ok_y, region_y = mlparams.get_int("input-region-y")
    ok_w, region_w = mlparams.get_int("input-region-width")
    ok_h, region_h = mlparams.get_int("input-region-height")
    if not (ok_x and ok_y and ok_h and ok_w):
        raise KeyError("Missing input region width/height in mlparams")

    return (region_x, region_y, region_w, region_h), (tensor_w, tensor_h)


def get_output_tensors(mlframe: Any) -> list[np.ndarray]:
    """Return a single-batch frame's output tensors as float32, batch dim removed.

    Parameters
    ----------
    mlframe
        The ``mlframe`` an ``MLPostprocess`` handler receives.

    Returns
    -------
    list[np.ndarray]
        One array per output tensor, in model output order.

    Raises
    ------
    KeyError
        If a tensor does not have a leading batch dimension of 1.
    """
    tensors: list[np.ndarray] = []
    for index in range(mlframe.info.n_tensors):
        tensor = mlframe.get_tensor(index)
        if not (tensor.ndim == 3 or (tensor.ndim == 2 and tensor.shape[0] == 1)):
            raise KeyError(f"Unexpected tensor shape: {tensor.shape}")
        tensors.append(tensor[0].astype(np.float32))
    return tensors


def execute_pipeline(pipeline: Pipeline, stop_timeout: float = 15.0) -> None:
    """Run `pipeline` until EOS, stopping it gracefully on Ctrl+C.

    ``Pipeline.execute()`` blocks in a GLib main loop, where a plain SIGINT
    handler would not run until the loop returns. It runs on a worker thread
    instead, and Ctrl+C calls ``pipeline.stop()``, which sends EOS so sinks such
    as `add_file_sink` finalize their output.

    Parameters
    ----------
    pipeline
        A fully built and linked pipeline.
    stop_timeout
        Seconds to wait for the pipeline to drain after Ctrl+C.
    """
    interrupted = threading.Event()

    def _on_sigint(signum: int, frame: FrameType | None) -> None:
        interrupted.set()

    previous_handler = signal.signal(signal.SIGINT, _on_sigint)
    runner = threading.Thread(target=pipeline.execute, daemon=True)
    runner.start()
    try:
        while runner.is_alive() and not interrupted.is_set():
            runner.join(timeout=0.2)
        if interrupted.is_set() and runner.is_alive():
            pipeline.stop()
            runner.join(timeout=stop_timeout)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
