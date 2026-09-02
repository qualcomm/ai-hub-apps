# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import re
from importlib import import_module

import jiwer
from qai_hub_models.datasets.libri_speech import LibriSpeechDataset
from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper
from tqdm import tqdm

SAMPLE_RATE = 16000


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def evaluate_whisper_accuracy(
    model_size: str,
    num_samples: int,
    omit_on_device: bool = False,
    encoder_path: str | None = None,
    decoder_path: str | None = None,
) -> None:
    """
    Evaluate Word Error Rate (WER) on the LibriSpeech Dataset.
    By default, computes metric on both PyTorch and NPU.
    Prints the metrics to console.

    Parameters
    ----------
    model_size
        Which version of Whisper is being used (e.g., tiny, base, etc.)
    num_samples
        Number of samples of the dataset to use for evaluation
    omit_on_device
        If set, only runs accuracy on the PyTorch model
    encoder_path
        The path to the encoder onnx model. If not set, assumes the encoder
        is in the location specified by the README instructions.
    decoder_path
        The path to the decoder onnx model. If not set, assumes the decoder
        is in the location specified by the README instructions.
    """
    dataset = LibriSpeechDataset()
    whisper_module = import_module(f"qai_hub_models.models.whisper_{model_size}")
    model = whisper_module.Model.from_pretrained()
    torch_app = whisper_module.App(
        model.encoder, model.decoder, model.get_hf_whisper_version()
    )
    on_device_app = None
    if not omit_on_device:
        if encoder_path is None:
            encoder_path = f"build\\whisper_{model_size}_float\\precompiled\\qualcomm-snapdragon-x-elite\\HfWhisperEncoder\\model.onnx"
        if decoder_path is None:
            decoder_path = f"build\\whisper_{model_size}_float\\precompiled\\qualcomm-snapdragon-x-elite\\HfWhisperDecoder\\model.onnx"
        on_device_app = whisper_module.App(
            OnnxModelTorchWrapper.OnNPU(encoder_path),
            OnnxModelTorchWrapper.OnNPU(decoder_path),
            model.get_hf_whisper_version(),
        )

    torch_predictions = []
    on_device_predictions = []
    gt = []
    for i, (audio, gt_chars) in tqdm(enumerate(dataset), total=num_samples):
        if i == num_samples:
            break
        gt.append("".join(chr(int(i)) for i in gt_chars if int(i) != 0))
        text = torch_app.transcribe(audio.numpy(), SAMPLE_RATE)
        # Replace punctuation a space and multiple spaces with a single space
        torch_predictions.append(normalize_text(text))
        if on_device_app is not None:
            text = on_device_app.transcribe(audio.numpy(), SAMPLE_RATE)
            # Replace punctuation a space and multiple spaces with a single space
            on_device_predictions.append(normalize_text(text))

    torch_wer = jiwer.wer(gt, torch_predictions)
    print(f"Torch WER on LibriSpeech: {torch_wer:.2%}")
    if not omit_on_device:
        on_device_wer = jiwer.wer(gt, on_device_predictions)
        print(f"On-Device WER on LibriSpeech: {on_device_wer:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo"],
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--omit-on-device",
        action="store_true",
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        help="Encoder model path",
    )
    parser.add_argument(
        "--decoder-path",
        type=str,
        help="Decoder model path",
    )
    args = parser.parse_args()
    evaluate_whisper_accuracy(
        args.model_size,
        args.num_samples,
        args.omit_on_device,
        args.encoder_path,
        args.decoder_path,
    )


if __name__ == "__main__":
    main()
