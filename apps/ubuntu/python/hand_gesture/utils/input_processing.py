# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Tuple


def get_gstreamer_input_pipeline(
    video_source: str, video_source_size: Tuple[int, int]
) -> str:
    """
    Build a GStreamer pipeline string for reading frames from a given video source,
    converting them to RGB with a horizontal flip, and exposing them via an appsink.

    Parameters
    ----------
    video_source : str
        The left-hand side of the pipeline specifying the source element and its properties,
        e.g. `"v4l2src device=/dev/video0"` or `"filesrc location=video.mp4 ! decodebin"`.
        This string should be a valid GStreamer element (or sub-pipeline) ending with `!`
        not required here because this function appends it for you.
    video_source_size : tuple[int, int]
        The (width, height) of the incoming video frames expected from `video_source`.
        These are used to set the caps on both the NV12 and RGB segments.

    Returns
    -------
    str
        A GStreamer pipeline description string.

    Notes
    -----
    - Requires the `qtivtransform` element (part of Qualcomm/Hexagon/GStreamer plugins) if `qtiqmmfsrc` is used; ensure it
      is available in your GStreamer setup.
    """
    video_source_width, video_source_height = video_source_size

    if video_source.lower().strip().startswith("qtiqmmfsrc"):
        return (
            f"{video_source} ! "
            f"video/x-raw,width={video_source_width},height={video_source_height},framerate=60/1,format=NV12 ! "
            f"qtivtransform flip-horizontal=true ! video/x-raw,format=RGB,width={video_source_width},height={video_source_height} ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            "appsink name=appsink drop=true sync=false max-buffers=1 emit-signals=true"
        )

    return (
        f"{video_source} ! "
        "videoconvert ! videoscale ! "
        f"video/x-raw,width={video_source_width},height={video_source_height},format=RGB ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        f"appsink name=appsink drop=true sync=false max-buffers=1 emit-signals=true"
    )
