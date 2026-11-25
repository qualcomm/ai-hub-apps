# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.platform import get_current_device
from qai_hub_apps_utils.quantization import dequantize, quantize
from utils.input_processing import read_video_rgb, sample_clips, sample_video
from utils.model_metadata import ModelMetadata, load_model_metadata
from utils.postprocessing import load_labels, top_k_predictions
from utils.preprocessing import (
    clip_to_input,
    preprocess_frame,
    preprocess_frame_kinetics,
)


def _set_input(
    interpreter: Interpreter,
    detail: dict[str, Any],
    clip: np.ndarray,
) -> None:
    """Quantize (if needed) and feed the preprocessed clip into the model.

    Parameters
    ----------
    interpreter
        TFLite interpreter for the classifier.
    detail
        The single input tensor detail from interpreter.get_input_details().
    clip
        Preprocessed clip already in the model's declared layout.
    """
    if np.issubdtype(detail["dtype"], np.integer):
        input_val = quantize(
            clip,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        input_val = clip.astype(detail["dtype"])
    interpreter.set_tensor(detail["index"], input_val)


def _get_output(interpreter: Interpreter, detail: dict[str, Any]) -> np.ndarray:
    """Read the output tensor, dequantizing it if the model is quantized.

    Parameters
    ----------
    interpreter
        TFLite interpreter for the classifier.
    detail
        The single output tensor detail from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Flat per-class score array.
    """
    tensor = interpreter.get_tensor(detail["index"])
    if np.issubdtype(detail["dtype"], np.integer):
        tensor = dequantize(
            tensor,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    return tensor.reshape(-1)


def run_inference(
    clip: np.ndarray,
    interpreter: Interpreter,
    input_detail: dict[str, Any],
    output_detail: dict[str, Any],
) -> np.ndarray:
    """Run the classifier on one clip/frame and return per-class scores.

    Parameters
    ----------
    clip
        Preprocessed clip in the model's declared layout.
    interpreter
        TFLite interpreter for the classifier.
    input_detail
        The input tensor detail from interpreter.get_input_details().
    output_detail
        The output tensor detail from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        Per-class scores (softmax probabilities).
    """
    _set_input(interpreter, input_detail, clip)
    interpreter.invoke()
    return _get_output(interpreter, output_detail)


def draw_predictions(
    rgb_frame: np.ndarray, predictions: list[tuple[str, float]]
) -> None:
    """Overlay the top predictions on an RGB frame in place.

    Parameters
    ----------
    rgb_frame
        RGB frame of shape [H, W, 3], dtype uint8; modified in place.
    predictions
        (class name, score) pairs, highest score first.
    """
    for row, (name, score) in enumerate(predictions):
        text = f"{name}: {score * 100:.1f}%"
        origin = (12, 32 + row * 34)
        # Black outline for legibility over any background, then the label.
        cv2.putText(
            rgb_frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5
        )
        cv2.putText(
            rgb_frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )


def load_interpreter(args: argparse.Namespace, metadata: ModelMetadata) -> Interpreter:
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
        str(Path("models") / metadata.model_filename), experimental_delegates=[delegate]
    )
    interpreter.allocate_tensors()
    return interpreter


def classify_video(
    args: argparse.Namespace,
    metadata: ModelMetadata,
    interpreter: Interpreter,
    input_detail: dict[str, Any],
    output_detail: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Classify a video; returns averaged per-class scores and a preview frame.

    Each sampled clip is classified independently and the per-clip softmax
    probabilities are averaged.
    """
    crop_h, crop_w = metadata.crop_size
    # 112x112 Kinetics models expect 15 fps frames and a fixed-size resize;
    # anything else keeps its native rate and aspect ratio.
    is_kinetics_112 = crop_h == crop_w == C.KINETICS_SPATIAL_DIM
    video = read_video_rgb(
        args.video, target_fps=C.KINETICS_TARGET_FPS if is_kinetics_112 else None
    )

    num_frames = metadata.num_frames
    if args.num_clips > 1:
        clips = sample_clips(video, num_frames, args.num_clips)
    else:
        clips = [sample_video(video, num_frames)]

    score_sum = None
    for raw_clip in clips:
        frames = [
            preprocess_frame_kinetics(frame, crop_h, crop_w)
            if is_kinetics_112
            else preprocess_frame(frame, crop_h, crop_w)
            for frame in raw_clip
        ]
        clip = clip_to_input(frames, channels_first=metadata.is_channels_first)
        scores = run_inference(clip, interpreter, input_detail, output_detail)
        score_sum = scores if score_sum is None else score_sum + scores

    assert score_sum is not None
    return score_sum / len(clips), video[len(video) // 2]


def main(args: argparse.Namespace) -> None:
    if not args.hexagon_version:
        raise SystemExit(
            "Unknown Hexagon version for this device. "
            "Pass it with --hexagon-version <e.g. v73>."
        )
    if args.num_clips < 1:
        raise SystemExit("--num-clips must be at least 1.")

    models_dir = Path("models")
    metadata = load_model_metadata(models_dir)
    labels = load_labels(models_dir, metadata)

    interpreter = load_interpreter(args, metadata)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    scores, preview_frame = classify_video(
        args, metadata, interpreter, input_detail, output_detail
    )
    predictions = top_k_predictions(scores, labels, args.top_k)

    for name, score in predictions:
        print(f"{name}: {score * 100:.1f}%")

    if args.output:
        annotated = preview_frame.copy()
        draw_predictions(annotated, predictions)
        cv2.imwrite(args.output, annotated[..., ::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Classification")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video to classify",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Write an annotated frame from the video to this path",
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=C.DEFAULT_TOP_K,
        help=f"Number of top predictions to overlay, default {C.DEFAULT_TOP_K}",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=C.DEFAULT_NUM_CLIPS,
        help="Clips to sample from the video, averaging their scores. "
        f"Default {C.DEFAULT_NUM_CLIPS}; raising it to 5 samples the video more "
        "densely, which is more accurate but slower.",
    )

    args = parser.parse_args()
    main(args)
