# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.quantization import dequantize, quantize
from utils.audio_processing import (
    chunk_and_resample_audio,
    load_audiofile,
    wav_to_logmel_patches,
)
from utils.postprocessing import load_class_labels, top_k_labels


def _set_input(
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    patch: np.ndarray,
) -> None:
    """Quantize (if needed) and feed one log-mel patch into the model.

    Parameters
    ----------
    interpreter
        TFLite interpreter for YamNet.
    input_details
        Input tensor details from interpreter.get_input_details().
    patch
        Log-mel patch of shape [1, 1, 96, 64], dtype float32.
    """
    detail = input_details[0]
    if np.issubdtype(detail["dtype"], np.integer):
        input_val = quantize(
            patch,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        input_val = patch.astype(detail["dtype"])
    interpreter.set_tensor(detail["index"], input_val)


def _get_output(
    interpreter: Interpreter,
    detail: dict[str, Any],
) -> np.ndarray:
    """Read one output tensor, dequantizing it if the model is quantized.

    Parameters
    ----------
    interpreter
        TFLite interpreter for YamNet.
    detail
        A single entry from interpreter.get_output_details().

    Returns
    -------
    np.ndarray
        The output tensor as float32.
    """
    tensor = interpreter.get_tensor(detail["index"])
    if np.issubdtype(detail["dtype"], np.integer):
        tensor = dequantize(
            tensor,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    return tensor


def classify_audio(
    audio_file: str,
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    output_details: list[dict[str, Any]],
    top_k: int,
) -> list[str]:
    """Run YamNet on an audio file and return the top-k class names.

    Parameters
    ----------
    audio_file
        Path to the input audio file.
    interpreter
        TFLite interpreter for YamNet.
    input_details
        Input tensor details from interpreter.get_input_details().
    output_details
        Output tensor details from interpreter.get_output_details().
    top_k
        Number of top predictions to return.

    Returns
    -------
    list[str]
        The top-``top_k`` AudioSet class names, highest score first.
    """
    audio, sample_rate = load_audiofile(audio_file)

    per_patch_scores = []
    for chunk in chunk_and_resample_audio(audio, sample_rate):
        for patch in wav_to_logmel_patches(chunk):
            _set_input(interpreter, input_details, patch[np.newaxis, ...])
            interpreter.invoke()
            scores = _get_output(interpreter, output_details[0])
            per_patch_scores.append(scores.reshape(-1))

    if not per_patch_scores:
        raise RuntimeError("Audio was too short to produce a single model patch.")

    # Average scores across all patches to get one prediction for the clip.
    mean_scores = np.mean(np.stack(per_patch_scores), axis=0)
    labels = load_class_labels(f"models/{C.LABELS_FILENAME}")
    return top_k_labels(mean_scores, labels, top_k)


def main(args: argparse.Namespace) -> None:
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

    interpreter = Interpreter("models/yamnet.tflite", experimental_delegates=[delegate])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = classify_audio(
        args.audio_file, interpreter, input_details, output_details, args.top_k
    )
    print(f"Top {args.top_k} predictions: {' | '.join(predictions)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YamNet Audio Classification")
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to the input audio file to classify",
    )
    parser.add_argument(
        "--qairt-path",
        type=Path,
        required=True,
        help="Path to QAIRT SDK root",
    )
    parser.add_argument(
        "--hexagon-version",
        type=str,
        default="v73",
        help="Hexagon version of the device, e.g. v73, default v73",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to report, default 5",
    )

    args = parser.parse_args()
    main(args)
