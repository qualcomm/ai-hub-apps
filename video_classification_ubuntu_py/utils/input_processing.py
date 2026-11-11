# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np


def read_video_rgb(path: str, target_fps: int | None = None) -> np.ndarray:
    """Decode a whole video as RGB frames, optionally resampled to a frame rate.

    Parameters
    ----------
    path
        Path to the video file.
    target_fps
        Frame rate to resample to. ``None`` keeps the video's native rate.

    Returns
    -------
    np.ndarray
        Frames of shape [T, H, W, 3], dtype uint8.
    """
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not read video: {path}")
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []
        while True:
            ok, bgr = capture.read()
            if not ok or bgr is None:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise FileNotFoundError(f"Could not read video: {path}")

    video = np.stack(frames, axis=0)
    if target_fps is None or native_fps <= 0 or abs(native_fps - target_fps) < 0.5:
        return video

    step = native_fps / target_fps
    num_out = max(1, int(len(video) / step))
    indices = np.minimum(
        (np.arange(num_out) * step).round().astype(int), len(video) - 1
    )
    return video[indices]


def sample_video(video: np.ndarray, num_frames: int) -> np.ndarray:
    """Stride ``num_frames`` frames across the whole video.

    Short videos are padded by repeating the last frame.

    Parameters
    ----------
    video
        Frames of shape [T, H, W, 3].
    num_frames
        Number of frames the model consumes.

    Returns
    -------
    np.ndarray
        Frames of shape [num_frames, H, W, 3].
    """
    total = len(video)
    if total == 0:
        raise ValueError("Video has no frames.")
    if total <= num_frames:
        pad = np.repeat(video[-1:], num_frames - total, axis=0)
        return np.concatenate([video, pad], axis=0)
    frame_rate = max(1, total // num_frames)
    indices = np.minimum(np.arange(num_frames) * frame_rate, total - 1)
    return video[indices]


def sample_clips(
    video: np.ndarray, num_frames: int, num_clips: int
) -> list[np.ndarray]:
    """Take ``num_clips`` uniformly spaced windows of consecutive frames.

    Clip starts are ``linspace(0, num_windows - 1, num_clips)`` floored, over
    ``num_windows = max(1, total - num_frames + 1)``, so the clips are spread
    evenly across the video.

    Parameters
    ----------
    video
        Frames of shape [T, H, W, 3].
    num_frames
        Number of consecutive frames per clip.
    num_clips
        Number of temporal clips to extract.

    Returns
    -------
    list[np.ndarray]
        ``num_clips`` arrays, each of shape [num_frames, H, W, 3].
    """
    if len(video) == 0:
        raise ValueError("Video has no frames.")
    num_windows = max(1, len(video) - num_frames + 1)
    starts = np.floor(np.linspace(0, num_windows - 1, num=num_clips)).astype(int)

    clips = []
    for start in starts:
        clip = video[start : start + num_frames]
        if len(clip) < num_frames:
            pad = np.repeat(clip[-1:], num_frames - len(clip), axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        clips.append(clip)
    return clips
