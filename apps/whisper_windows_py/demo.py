# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
from datetime import datetime

import sounddevice as sd
from qai_hub_apps_utils.input_devices import get_default_audio_device
from utils.model import WhisperApp, load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="error",
    )
    # With neither of these, the default input device is streamed from.
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Audio file path or URL. Omit to stream from a microphone instead.",
    )
    input_group.add_argument(
        "--stream-audio-device",
        type=int,
        default=None,
        help="Audio device (number) to stream from, defaults to the system's default input device.",
    )
    parser.add_argument(
        "--stream-audio-chunk-size",
        type=int,
        default=10,
        help="For audio streaming, the number of seconds to record between each transcription attempt. A minimum of around 10 seconds is recommended for best accuracy.",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Pass this to list audio devices and exit.",
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        default="models\\encoder.onnx",
        help="Encoder model path",
    )
    parser.add_argument(
        "--decoder-path",
        type=str,
        default="models\\decoder.onnx",
        help="Decoder model path",
    )
    args = parser.parse_args()

    if args.list_audio_devices:
        print(sd.query_devices())
        return

    # Resolve the microphone before the slow model load so a host with no
    # capture device fails immediately.
    stream_device: int | None = None
    if args.audio_file is None:
        if args.stream_audio_device is not None:
            stream_device = args.stream_audio_device
        else:
            try:
                stream_device = get_default_audio_device()
            except RuntimeError as error:
                raise SystemExit(
                    f"{error} List the available devices with --list-audio-devices "
                    "and pass one with --stream-audio-device <n>, or transcribe a "
                    "file with --audio-file <path>."
                ) from error

    print("Loading model...")
    model = load_model(
        args.encoder_path,
        args.decoder_path,
    )

    app = WhisperApp(model)

    if stream_device is not None:
        print(f"Streaming from audio device {stream_device}.")
        app.stream(stream_device, args.stream_audio_chunk_size)
    else:
        # Perform transcription
        print("Before transcription: " + str(datetime.now().astimezone()))
        transcription = app.transcribe(args.audio_file)
        print(f"Transcription: {transcription}")
        print("After transcription: " + str(datetime.now().astimezone()))


if __name__ == "__main__":
    main()
