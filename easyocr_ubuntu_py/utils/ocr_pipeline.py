# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Two-stage OCR pipeline (detect -> group -> recognize).

Written against a detector and a recognizer callable, so the TFLite/QNN details
live in ``main.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
from qai_hub_apps_utils.image_processing import resize_pad

import utils.constants as C
from utils.craft_decode import get_det_boxes
from utils.ctc_decode import CTCGreedyDecoder
from utils.text_grouping import diff, four_point_transform, group_text_box

# A detection is (x_min, y_min, x_max, y_max, text, confidence).
Detection = tuple[int, int, int, int, str, float]

# Callable that maps a preprocessed input array to the model's raw output array.
ModelFn = Callable[[np.ndarray], np.ndarray]


class RunStats:
    """Timing/count info from the most recent `OCRPipeline.run()` call.

    Populated by `run()` for display in the app's "Info" panel -- not used
    internally by the pipeline itself.
    """

    def __init__(self) -> None:
        self.detector_ms: float = 0.0
        self.recognizer_ms_total: float = 0.0
        self.num_boxes_detected: int = 0
        self.num_boxes_recognized: int = 0

    @property
    def recognizer_ms_avg(self) -> float:
        if self.num_boxes_recognized == 0:
            return 0.0
        return self.recognizer_ms_total / self.num_boxes_recognized

    @property
    def total_ms(self) -> float:
        return self.detector_ms + self.recognizer_ms_total


class OCRPipeline:
    def __init__(
        self,
        detector_fn: ModelFn,
        recognizer_fns: Sequence[ModelFn],
        detector_hw: tuple[int, int],
        recognizer_hw: tuple[int, int],
        charset: str = C.CHARSET,
        max_recognitions: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        detector_fn
            Maps a uint8 RGB image [H, W, 3] (detector input size) to the detector
            score map.
        recognizer_fns
            One or more callables, each mapping a float32 grayscale crop
            [H, W, 1] in [0, 1] (recognizer input size) to per-timestep class
            logits [T, num_classes]. Each must wrap its OWN interpreter (an
            interpreter is not thread-safe); with N callables, N boxes are
            recognized in parallel.
        detector_hw
            (height, width) of the detector input.
        recognizer_hw
            (height, width) of the recognizer input.
        charset
            Recognizer character set (order significant; blank is class 0).
        max_recognitions
            Run the (expensive, once-per-box) recognizer on at most this many
            detector boxes per frame; extra boxes are dropped before
            recognition, not just before display. `None` recognizes every
            detected box.
        """
        self.detector_fn = detector_fn
        self.recognizer_fns = list(recognizer_fns)
        self._pool = (
            ThreadPoolExecutor(len(self.recognizer_fns))
            if len(self.recognizer_fns) > 1
            else None
        )
        self.detector_hw = detector_hw
        self.recognizer_hw = recognizer_hw
        self.decoder = CTCGreedyDecoder(charset)
        self.max_recognitions = max_recognitions
        self.stats = RunStats()

    def run(self, rgb_frame: np.ndarray) -> list[Detection]:
        """Detect and recognize text in an RGB uint8 frame.

        Recognizes at most `self.max_recognitions` of the detected boxes (see
        `__init__`), not necessarily all boxes the detector found. Timing/count
        info from this call is left in `self.stats` for display.
        """
        self.stats = RunStats()

        detect_start = time.perf_counter()
        boxes, scores = self._detect(rgb_frame)
        self.stats.detector_ms = (time.perf_counter() - detect_start) * 1000
        self.stats.num_boxes_detected = len(boxes)
        if not boxes:
            return []

        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        horizontal_list, free_list = group_text_box(
            boxes,
            slope_ths=C.SLOPE_THS,
            ycenter_ths=C.YCENTER_THS,
            height_ths=C.HEIGHT_THS,
            width_ths=C.WIDTH_THS,
            add_margin=C.ADD_MARGIN,
        )

        # Minimum-size filtering (EasyOCR DETECTOR_ARGS["min_size"]).
        horizontal_list = [
            b for b in horizontal_list if max(b[1] - b[0], b[3] - b[2]) > C.MIN_SIZE
        ]
        free_list = [
            b
            for b in free_list
            if max(diff([c[0] for c in b]), diff([c[1] for c in b])) > C.MIN_SIZE
        ]

        # Rank every line (horizontal and slanted together) by its best detector
        # score, then keep the top N.
        n_horizontal = len(horizontal_list)
        lines = [*horizontal_list, *free_list]
        if self.max_recognitions is not None and len(lines) > self.max_recognitions:
            line_scores = _line_scores(lines, boxes, scores)
            ranked = sorted(
                range(len(lines)), key=lambda i: line_scores[i], reverse=True
            )
            keep = set(ranked[: self.max_recognitions])
            horizontal_list = [b for i, b in enumerate(horizontal_list) if i in keep]
            free_list = [
                b for i, b in enumerate(free_list, start=n_horizontal) if i in keep
            ]

        tasks: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []

        # Slanted (free) boxes: warp to a rectangle before recognition.
        for box in free_list:
            rect = np.array(box, dtype="float32")
            crop = four_point_transform(gray, rect)
            if 0 in crop.shape:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            tasks.append(
                (crop, (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
            )

        # Axis-aligned boxes: crop directly.
        for line in horizontal_list:
            x_min = max(0, int(line[0]))
            x_max = int(min(line[1], gray.shape[1]))
            y_min = max(0, int(line[2]))
            y_max = int(min(line[3], gray.shape[0]))
            if y_max - y_min <= 0 or x_max - x_min <= 0:
                continue
            tasks.append((gray[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)))

        return self._recognize_all(tasks)

    def _detect(self, rgb_frame: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        """Run the detector; return flat 8-element boxes (image pixels) + scores."""
        padded, scale, (pad_left, pad_top) = resize_pad(rgb_frame, self.detector_hw)

        score_map = self.detector_fn(padded)
        score_text, score_link = _split_score_map(score_map)

        raw_boxes, scores = get_det_boxes(
            score_text,
            score_link,
            text_threshold=C.TEXT_THRESHOLD,
            link_threshold=C.LINK_THRESHOLD,
            low_text=C.LOW_TEXT,
        )

        # Map score-map coords -> detector-input coords (x net ratio) -> original
        # image coords (remove padding, undo resize scale).
        boxes: list[np.ndarray] = []
        for box in raw_boxes:
            mapped = box.astype(np.float32)
            mapped[:, 0] = (mapped[:, 0] * C.DETECTOR_NET_RATIO - pad_left) / scale
            mapped[:, 1] = (mapped[:, 1] * C.DETECTOR_NET_RATIO - pad_top) / scale
            boxes.append(mapped.astype(np.int32).reshape(-1))
        return boxes, scores

    def _recognize_all(
        self, tasks: list[tuple[np.ndarray, tuple[int, int, int, int]]]
    ) -> list[Detection]:
        """Recognize every crop, spread across the recognizers; keep box order.

        `stats.recognizer_ms_total` is the wall-clock time of this whole stage, so
        with several recognizers the per-box average reflects the real throughput.
        """
        start = time.perf_counter()
        workers = min(len(self.recognizer_fns), len(tasks))
        results: list[Detection | None] = [None] * len(tasks)

        def work(worker: int) -> None:
            for idx in range(worker, len(tasks), max(workers, 1)):
                crop, xyxy = tasks[idx]
                results[idx] = self._recognize(self.recognizer_fns[worker], crop, xyxy)

        if self._pool is not None and workers > 1:
            list(self._pool.map(work, range(workers)))
        else:
            work(0)

        self.stats.recognizer_ms_total = (time.perf_counter() - start) * 1000
        self.stats.num_boxes_recognized = len(tasks)
        return [r for r in results if r is not None]

    def _recognize(
        self,
        recognizer_fn: ModelFn,
        gray_crop: np.ndarray,
        xyxy: tuple[int, int, int, int],
    ) -> Detection | None:
        logits = recognizer_fn(self._prepare_recognizer_input(gray_crop))
        probs = _softmax_last_axis(_squeeze_to_2d(logits))
        text, confidence = self.decoder.decode(probs)

        text = text.strip()
        # The recognizer can hallucinate a trailing "]" / "|" over right padding.
        if text and text[-1] in ("]", "|"):
            text = text[:-1].strip()
        if not text:
            return None
        return (*xyxy, text, float(confidence))

    def _prepare_recognizer_input(self, gray_crop: np.ndarray) -> np.ndarray:
        """Resize a grayscale crop to recognizer input, left-aligned + right-padded.

        Mirrors EasyOCR's recognizer preprocessing: preserve aspect by height,
        left-align, and pad the remaining width with the crop's background
        (top-left pixel) value.
        """
        target_h, target_w = self.recognizer_hw
        h, w = gray_crop.shape[:2]
        pad_value = int(gray_crop[0, 0]) if gray_crop.size else 0

        new_h = target_h
        new_w = max(1, min(target_w, round(w * (target_h / h))))
        resized = cv2.resize(gray_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_h, target_w), pad_value, dtype=np.uint8)
        canvas[:, :new_w] = resized

        return (canvas.astype(np.float32) / 255.0)[..., None]


def _word_center(word: np.ndarray) -> tuple[float, float]:
    """Center of a flat 8-element word box [x1, y1, ..., x4, y4]."""
    return float(word[0::2].mean()), float(word[1::2].mean())


def _line_center(line: Any) -> tuple[float, float]:
    """Center of a line box: [x_min, x_max, y_min, y_max] or 4 corner points."""
    if len(line) == 4 and not hasattr(line[0], "__len__"):
        return (line[0] + line[1]) / 2, (line[2] + line[3]) / 2
    return (
        sum(p[0] for p in line) / len(line),
        sum(p[1] for p in line) / len(line),
    )


def _line_scores(
    lines: list[Any], word_boxes: list[np.ndarray], scores: list[float]
) -> list[float]:
    """Best detector score of each line, from the word boxes merged into it.

    `group_text_box` merges word boxes into lines but drops which words went
    where, so each word is assigned to the line whose center is nearest to the
    word's center, and a line keeps the highest score among its words.
    """
    centers = [_line_center(line) for line in lines]
    best = [0.0] * len(lines)
    if not centers:
        return best
    for word, score in zip(word_boxes, scores, strict=True):
        wx, wy = _word_center(word)
        nearest = min(
            range(len(centers)),
            key=lambda i: (centers[i][0] - wx) ** 2 + (centers[i][1] - wy) ** 2,
        )
        best[nearest] = max(best[nearest], score)
    return best


def _split_score_map(score_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (text score map, link score map) from the detector output.

    Handles a leading batch dim and both channels-last [H, W, 2] and
    channels-first [2, H, W] layouts.
    """
    arr = np.asarray(score_map)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected detector output shape: {score_map.shape}")
    if arr.shape[-1] == 2:  # channels-last [H, W, 2]
        return arr[:, :, 0], arr[:, :, 1]
    if arr.shape[0] == 2:  # channels-first [2, H, W]
        return arr[0], arr[1]
    raise ValueError(f"Cannot locate 2-channel score map in shape {score_map.shape}")


def _squeeze_to_2d(logits: np.ndarray) -> np.ndarray:
    """Drop a leading batch dim so recognizer logits are [T, num_classes]."""
    arr = np.asarray(logits)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected recognizer output shape: {logits.shape}")
    return arr


def _softmax_last_axis(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
