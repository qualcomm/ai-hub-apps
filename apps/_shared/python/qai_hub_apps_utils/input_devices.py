# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Helpers for selecting live capture devices on the host."""

import fcntl
import glob
import os
import struct

# VIDIOC_QUERYCAP and the subset of struct v4l2_capability flags needed to tell a
# capture node from the metadata/output nodes the same camera also exposes.
_VIDIOC_QUERYCAP = 0x80685600
_QUERYCAP_FORMAT = "16s32s32sIII12x"
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
_V4L2_CAP_DEVICE_CAPS = 0x80000000


def get_default_audio_device() -> int:
    """Return the microphone to capture from, as a PortAudio device index.

    The index is what ``sounddevice`` accepts as its ``device`` argument and
    what ``sounddevice.query_devices()`` enumerates; it is not an ALSA card
    number.

    Returns
    -------
    int
        The host's default input device, or the first device with an input
        channel when the host has no usable default set.

    Raises
    ------
    RuntimeError
        If the host has no audio capture device.
    """
    import sounddevice as sd

    try:
        index = sd.default.device[0]
        if (
            index is None
            or index < 0
            or sd.query_devices(index)["max_input_channels"] < 1
        ):
            index = next(
                i
                for i, device in enumerate(sd.query_devices())
                if device["max_input_channels"] > 0
            )
    except (StopIteration, sd.PortAudioError, ValueError) as error:
        raise RuntimeError("No audio capture device found on this host.") from error
    return int(index)


def _node_capabilities(path: str) -> int:
    """Return the V4L2 capability flags that apply to one /dev/video* node.

    Parameters
    ----------
    path
        Path to the V4L2 node to query.

    Returns
    -------
    int
        The node's own capabilities when the driver reports them, else the
        capabilities of the physical device as a whole.
    """
    # O_NONBLOCK so querying a node another process is streaming from cannot hang.
    node = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        queried = fcntl.ioctl(
            node, _VIDIOC_QUERYCAP, bytes(struct.calcsize(_QUERYCAP_FORMAT))
        )
    finally:
        os.close(node)
    *_, capabilities, device_caps = struct.unpack(_QUERYCAP_FORMAT, queried)
    # `capabilities` is the union across every node of the physical device, so a
    # metadata node reports VIDEO_CAPTURE there too; `device_caps` is per-node.
    if capabilities & _V4L2_CAP_DEVICE_CAPS:
        return int(device_caps)
    return int(capabilities)


def get_default_video_device() -> str:
    """Return the camera to capture from, as a V4L2 device path.

    The path is what a V4L2 consumer takes directly, such as GStreamer's
    ``v4l2src device=`` or OpenCV's ``VideoCapture``. Only V4L2 nodes are
    considered.

    Returns
    -------
    str
        Path to the lowest-numbered /dev/video* node that can capture video.

    Raises
    ------
    RuntimeError
        If the host has no video capture device.
    """
    nodes = []
    for path in glob.glob("/dev/video*"):
        suffix = path.removeprefix("/dev/video")
        if suffix.isdigit():
            nodes.append((int(suffix), path))

    for _, path in sorted(nodes):
        try:
            capabilities = _node_capabilities(path)
        except OSError:
            # Not a queryable V4L2 node, or we cannot open it; try the next one.
            continue
        if capabilities & (_V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_VIDEO_CAPTURE_MPLANE):
            return path

    raise RuntimeError("No video capture device found on this host.")
