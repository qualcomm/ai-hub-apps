# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.input_devices import get_default_audio_device
from qai_hub_apps_utils.platform import get_current_device
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


def _patch_scores(
    chunk: np.ndarray,
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    output_details: list[dict[str, Any]],
) -> list[np.ndarray]:
    """Score every log-mel patch of one waveform chunk.

    Parameters
    ----------
    chunk
        Waveform of shape [1, num_samples], float32 at ``C.SAMPLE_RATE``.
    interpreter
        TFLite interpreter for YamNet.
    input_details
        Input tensor details from interpreter.get_input_details().
    output_details
        Output tensor details from interpreter.get_output_details().

    Returns
    -------
    list[np.ndarray]
        One flat per-class score array per patch. Empty if the chunk is shorter
        than a single patch window.
    """
    scores = []
    for patch in wav_to_logmel_patches(chunk):
        _set_input(interpreter, input_details, patch[np.newaxis, ...])
        interpreter.invoke()
        scores.append(_get_output(interpreter, output_details[0]).reshape(-1))
    return scores


def classify_stream(
    device: int,
    interpreter: Interpreter,
    input_details: list[dict[str, Any]],
    output_details: list[dict[str, Any]],
    top_k: int,
) -> None:
    """Classify audio from a microphone until interrupted, printing each window.

    Parameters
    ----------
    device
        Audio device index to capture from (see ``sounddevice.query_devices()``).
    interpreter
        TFLite interpreter for YamNet.
    input_details
        Input tensor details from interpreter.get_input_details().
    output_details
        Output tensor details from interpreter.get_output_details().
    top_k
        Number of top predictions to report per window.
    """
    # Capture at the model's rate when the device supports it, else at the
    # device's own rate and let chunk_and_resample_audio resample each window.
    samplerate = C.SAMPLE_RATE
    try:
        sd.check_input_settings(device=device, samplerate=samplerate, channels=1)
    except sd.PortAudioError:
        samplerate = int(sd.query_devices(device)["default_samplerate"])

    block_frames = int(samplerate * C.CHUNK_LENGTH)
    labels = load_class_labels(f"models/{C.LABELS_FILENAME}")

    print(f"Listening on audio device {device}. Press Ctrl+C to stop.", flush=True)
    with sd.InputStream(
        device=device,
        channels=1,
        samplerate=samplerate,
        dtype="float32",
        blocksize=block_frames,
    ) as stream:
        try:
            while True:
                frames, overflowed = stream.read(block_frames)
                if overflowed:
                    # Samples were dropped; this window is not worth scoring.
                    continue
                for chunk in chunk_and_resample_audio(frames.T, samplerate):
                    scores = _patch_scores(
                        chunk, interpreter, input_details, output_details
                    )
                    if not scores:
                        continue
                    mean_scores = np.mean(np.stack(scores), axis=0)
                    predictions = top_k_labels(mean_scores, labels, top_k)
                    print(f"{' | '.join(predictions)}", flush=True)
        except KeyboardInterrupt:
            print()


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
        per_patch_scores.extend(
            _patch_scores(chunk, interpreter, input_details, output_details)
        )

    if not per_patch_scores:
        raise RuntimeError("Audio was too short to produce a single model patch.")

    # Average scores across all patches to get one prediction for the clip.
    mean_scores = np.mean(np.stack(per_patch_scores), axis=0)
    labels = load_class_labels(f"models/{C.LABELS_FILENAME}")
    return top_k_labels(mean_scores, labels, top_k)


def main(args: argparse.Namespace) -> None:
    if args.list_audio_devices:
        print(sd.query_devices())
        return

    if not args.hexagon_version:
        raise SystemExit(
            "Unknown Hexagon version for this device. "
            "Pass it with --hexagon-version <e.g. v73>."
        )

    # Resolve the microphone before the model load so a host with no capture
    # device fails immediately.
    stream_device: int | None = None
    if args.audio_file is None:
        if args.audio_device is not None:
            stream_device = args.audio_device
        else:
            try:
                stream_device = get_default_audio_device()
            except RuntimeError as error:
                raise SystemExit(
                    f"{error} List the available devices with --list-audio-devices "
                    "and pass one with --audio-device <n>, or classify a file with "
                    "--audio-file <path>."
                ) from error

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

    if stream_device is not None:
        classify_stream(
            stream_device, interpreter, input_details, output_details, args.top_k
        )
        return

    predictions = classify_audio(
        args.audio_file, interpreter, input_details, output_details, args.top_k
    )
    print(f"Top {args.top_k} predictions: {' | '.join(predictions)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YamNet Audio Classification")
    # With neither of these, the default input device is streamed from.
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to an audio file to classify. Omit to stream from a microphone.",
    )
    input_group.add_argument(
        "--audio-device",
        type=int,
        default=None,
        help="Audio device (number) to stream from, defaults to the system's "
        "default input device",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List the available audio devices and exit",
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
        default=5,
        help="Number of top predictions to report, default 5",
    )

    args = parser.parse_args()
    main(args)
