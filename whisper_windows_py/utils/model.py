# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Autoregressive Whisper decode over the encoder/decoder QNN ONNX graphs.

A real transformers config/tokenizer/feature-extractor drives the loop; the
encoder/decoder run via ONNX Runtime's QNN Execution Provider (see
qai_hub_apps_utils.onnxruntime_qnn.open_qnn_session). Supports any Whisper model
variant available on HuggingFace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnxruntime as ort
import sounddevice as sd
from qai_hub_apps_utils.onnxruntime_qnn import open_qnn_session
from scipy.signal import resample_poly
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperTokenizer

SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
MEAN_DECODE_LEN = 200
MASK_NEG = -100.0

_ORT_TYPE_TO_NP_DTYPE: dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
}


def _run_onnx(
    session: ort.InferenceSession, *args: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Run ``session`` on positional arrays (in the graph's input order).

    Each input is cast to the graph's declared dtype; all outputs are returned
    as a tuple in declared order.
    """
    inputs = session.get_inputs()
    feed: dict[str, np.ndarray] = {
        i.name: arg.astype(_ORT_TYPE_TO_NP_DTYPE[i.type], copy=False)
        for i, arg in zip(inputs, args, strict=False)
    }
    output_names = [o.name for o in session.get_outputs()]
    return tuple(session.run(output_names, feed))


@dataclass
class WhisperModel:
    """The loaded Whisper encoder/decoder QNN sessions plus config/tokenizer state."""

    encoder: ort.InferenceSession
    decoder: ort.InferenceSession
    config: WhisperConfig
    tokenizer: WhisperTokenizer
    feature_extractor: WhisperFeatureExtractor
    clip_segment_tokens: set[int]


def load_model(
    encoder_path: str, decoder_path: str, model_size: str = "base"
) -> WhisperModel:
    """Open the encoder/decoder on the QNN EP and load the HF config/tokenizer.

    Parameters
    ----------
    encoder_path
        Path to the encoder ONNX model.
    decoder_path
        Path to the decoder ONNX model.
    model_size
        Whisper model size, used to select the HuggingFace model id
        ``openai/whisper-{model_size}`` (e.g. ``base``, ``small``).

    Returns
    -------
    WhisperModel
        The loaded sessions and transformers config/tokenizer/feature-extractor.
    """
    hf_model_id = f"openai/whisper-{model_size}"
    config = WhisperConfig.from_pretrained(hf_model_id)
    config.return_dict = False
    config.tie_word_embeddings = False
    tokenizer = WhisperTokenizer.from_pretrained(hf_model_id)
    return WhisperModel(
        encoder=open_qnn_session(encoder_path),
        decoder=open_qnn_session(decoder_path),
        config=config,
        tokenizer=tokenizer,
        feature_extractor=WhisperFeatureExtractor.from_pretrained(hf_model_id),
        clip_segment_tokens=set(tokenizer.all_special_ids),
    )


class WhisperApp:
    """Transcribe audio with a loaded Whisper encoder/decoder.

    Wraps a :class:`WhisperModel` and drives the autoregressive decode loop,
    exposing file/array transcription and real-time microphone streaming.
    """

    def __init__(self, model: WhisperModel) -> None:
        """Wrap a model returned by :func:`load_model`.

        Parameters
        ----------
        model
            A model returned by ``load_model``.
        """
        self.model = model

    def transcribe(
        self, audio: np.ndarray | str, audio_sample_rate: int | None = None
    ) -> str:
        """Transcribe audio to text.

        Parameters
        ----------
        audio
            Path to an audio file (str), or a raw audio array of shape (# samples).
        audio_sample_rate
            Sample rate of ``audio`` in samples/sec. Required if ``audio`` is an
            array; ignored (derived from the file) if ``audio`` is a path.

        Returns
        -------
        str
            The transcribed text.
        """
        tokens = self.transcribe_tokens(audio, audio_sample_rate)
        return self.model.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def transcribe_tokens(
        self, audio: np.ndarray | str, audio_sample_rate: int | None = None
    ) -> list[int]:
        """Transcribe audio to token ids.

        Parameters
        ----------
        audio
            Path to an audio file (str), or a raw audio array of shape (# samples).
        audio_sample_rate
            Sample rate of ``audio`` in samples/sec. Required if ``audio`` is an
            array; ignored (derived from the file) if ``audio`` is a path.

        Returns
        -------
        list[int]
            The transcribed token ids.
        """
        if isinstance(audio, str):
            import audio2numpy as a2n  # requires ffmpeg on the host machine

            audio, audio_sample_rate = a2n.audio_from_file(audio)
            if isinstance(audio, np.ndarray) and audio.ndim == 2:
                # Audio is multi-channel (e.g., stereo); collapse to single.
                audio = audio.mean(-1)

        assert audio_sample_rate is not None
        assert isinstance(audio, np.ndarray)

        out_tokens: list[int] = []
        for chunk in self.chunk_and_resample_audio(audio, audio_sample_rate):
            out_tokens.extend(self._transcribe_single_chunk(chunk))
        return out_tokens

    def _transcribe_single_chunk(self, audio: np.ndarray) -> list[int]:
        """Transcribe a single audio chunk (<= CHUNK_LENGTH seconds) to token ids.

        The audio must already be at SAMPLE_RATE.
        """
        config = self.model.config
        input_features = self.model.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
        )["input_features"]

        # Encoder returns a flat tuple (k0, v0, k1, v1, ...) of cross-attention kv cache.
        kv_cache_cross_flat = _run_onnx(self.model.encoder, input_features)
        kv_cache_cross = tuple(
            kv_cache_cross_flat[i : i + 2]
            for i in range(0, len(kv_cache_cross_flat), 2)
        )

        sot = config.decoder_start_token_id
        num_decoder_blocks = config.decoder_layers
        attention_dim = config.d_model
        num_decoder_heads = config.decoder_attention_heads
        eot = config.eos_token_id

        output_ids = np.array([[sot]])  # Start of transcript
        output_logits = []
        output_length = output_ids.shape[1]

        position_ids = np.array([0], dtype=np.int32)
        attention_mask = np.full(
            (1, 1, 1, MEAN_DECODE_LEN),
            MASK_NEG,
            dtype=np.float32,
        )

        # init kv_cache_self
        k_cache_self = np.zeros(
            (
                num_decoder_heads,
                1,
                attention_dim // num_decoder_heads,
                MEAN_DECODE_LEN - 1,
            ),
            dtype=np.float32,
        )
        v_cache_self = np.zeros(
            (
                num_decoder_heads,
                1,
                MEAN_DECODE_LEN - 1,
                attention_dim // num_decoder_heads,
            ),
            dtype=np.float32,
        )
        kv_cache_self: tuple[tuple[np.ndarray, ...], ...] = tuple(
            (k_cache_self, v_cache_self) for _ in range(num_decoder_blocks)
        )

        for n in range(MEAN_DECODE_LEN - 1):
            input_ids = output_ids[:, n : n + 1].astype(np.int32)

            attention_mask[:, :, :, MEAN_DECODE_LEN - n - 1] = 0.0

            flattened_kv_cache_self = tuple(
                item for sublist in kv_cache_self for item in sublist
            )
            flattened_kv_cache_cross = tuple(
                item for sublist in kv_cache_cross for item in sublist
            )

            decoder_output = _run_onnx(
                self.model.decoder,
                input_ids,
                attention_mask,
                *flattened_kv_cache_self,
                *flattened_kv_cache_cross,
                position_ids,
            )
            logits = decoder_output[0]
            kv_cache_self = tuple(
                decoder_output[i : i + 2] for i in range(1, len(decoder_output), 2)
            )

            output_logits.append(logits)

            output_id = np.argmax(logits, axis=1).reshape(1, 1)
            if len(output_logits) == (MEAN_DECODE_LEN - 1) or output_id == eot:
                output_ids = np.concatenate((output_ids, output_id), axis=-1)
                break
            if n >= output_length - 1:
                output_ids = np.concatenate((output_ids, output_id), axis=-1)

            position_ids += 1

        return output_ids[0].tolist()

    def stream(self, device: int = 2, audio_chunk_size_seconds: int = 5) -> None:
        """Stream audio from the given audio device and transcribe in real time.

        Parameters
        ----------
        device
            Audio device (see sounddevice.query_devices()).
        audio_chunk_size_seconds
            Number of seconds to record between each transcription attempt.
        """
        tokens: list[int] = []

        def callback(audio: np.ndarray, frames: int, time: Any, status: Any) -> None:
            nonlocal tokens
            curr_tokens = self.transcribe_tokens(audio.squeeze(-1), SAMPLE_RATE)
            tokens.extend(curr_tokens)

            if not curr_tokens:
                # This audio was empty, so it's safe to decode previous tokens.
                print(
                    self.model.tokenizer.decode(tokens, skip_special_tokens=True),
                    end="",
                    flush=True,
                )
                tokens = []
            else:
                split_start = 0
                decode_splits = []
                token_idx = 0
                # Every time 2 "clip segment tokens" (timestamp tokens)
                # appear in sequence, we're safe to decode the previous tokens.
                while token_idx < len(tokens):
                    if tokens[token_idx] in self.model.clip_segment_tokens:
                        next_non_clip_idx = token_idx + 1
                        while (
                            next_non_clip_idx < len(tokens)
                            and tokens[next_non_clip_idx]
                            in self.model.clip_segment_tokens
                        ):
                            next_non_clip_idx = next_non_clip_idx + 1

                        if next_non_clip_idx >= token_idx + 2:
                            split_end = token_idx + 1
                            if max(split_end - split_start, 0) > 0:
                                decode_splits.append((split_start, split_end))
                            split_start = next_non_clip_idx

                        token_idx = next_non_clip_idx + 1
                    else:
                        token_idx = token_idx + 1

                for split in decode_splits:
                    print(
                        self.model.tokenizer.decode(
                            tokens[split[0] : split[1]], skip_special_tokens=True
                        ),
                        end="",
                        flush=True,
                    )
                if split_start != 0:
                    tokens = tokens[split_start:]

        print("Listening...")
        print("Text can take up to 20 seconds before printing.")
        with sd.InputStream(
            device=device,
            channels=1,
            blocksize=audio_chunk_size_seconds * SAMPLE_RATE,
            callback=callback,
            samplerate=SAMPLE_RATE,
        ):
            while True:
                response = input("Press ctrl+c or q/Q to quit.\n")
                if response in ("q", "Q"):
                    break

    def chunk_and_resample_audio(
        self,
        audio: np.ndarray,
        audio_sample_rate: int,
        model_sample_rate: int = SAMPLE_RATE,
        model_chunk_seconds: int = CHUNK_LENGTH,
    ) -> list[np.ndarray]:
        """
        Chunk and resample audio data for model processing.

        Parameters
        ----------
        audio
            Raw audio numpy array of shape [# of samples]
        audio_sample_rate
            Sample rate of audio array, in samples / sec.
        model_sample_rate
            Sample rate (samples / sec) required to run Whisper. The audio file
            will be resampled to use this rate.
        model_chunk_seconds
            Split the audio in to N sequences of this many seconds.
            The final split may be shorter than this many seconds.

        Returns
        -------
        audio_chunks : list[np.ndarray]
            List of audio arrays, chunked into N arrays of model_chunk_seconds seconds.
        """
        if audio_sample_rate != model_sample_rate:
            audio = resample_poly(audio, model_sample_rate, audio_sample_rate)
            audio_sample_rate = model_sample_rate

        number_of_full_length_audio_chunks = (
            audio.shape[0] // audio_sample_rate // model_chunk_seconds
        )
        last_sample_in_full_length_audio_chunks = (
            audio_sample_rate * number_of_full_length_audio_chunks * model_chunk_seconds
        )

        if number_of_full_length_audio_chunks == 0:
            return [audio]

        return [
            *np.array_split(
                audio[:last_sample_in_full_length_audio_chunks],
                number_of_full_length_audio_chunks,
            ),
            audio[last_sample_in_full_length_audio_chunks:],
        ]
