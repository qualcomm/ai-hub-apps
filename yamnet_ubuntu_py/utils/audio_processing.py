# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Numpy log-mel front end for YamNet. The exported TFLite graph takes a
# (1, 1, 96, 64) log-mel patch, not a raw waveform, so preprocessing is done
# here.
# References:
#   - torch_audioset:
#     https://github.com/w-hc/torch_audioset (commit e8852c5)
#   - Google's original VGGish/YAMNet numpy mel front end that the math below
#     is ported from:
#     https://github.com/tensorflow/models/blob/master/research/audioset/vggish/mel_features.py

from __future__ import annotations

import numpy as np
import resampy
import soundfile as sf

import utils.constants as C


def load_audiofile(path: str) -> tuple[np.ndarray, int]:
    """Decode an audio file to a mono float32 waveform.

    Parameters
    ----------
    path
        Path to the input audio file (any format libsndfile/ffmpeg can read).

    Returns
    -------
    audio : np.ndarray
        Waveform of shape [1, num_samples], float32 in the range [-1, 1).
    sample_rate : int
        Sample rate of the decoded audio, in samples / second.
    """
    x, sample_rate = sf.read(path, dtype="int16", always_2d=True)
    x = x / 2**15
    x = x.T.astype(np.float32)
    # Average channels down to mono.
    if x.shape[0] > 1:
        x = np.mean(x, axis=0, keepdims=True)
    return x, sample_rate


def chunk_and_resample_audio(
    audio: np.ndarray,
    audio_sample_rate: int,
    model_sample_rate: int = C.SAMPLE_RATE,
    model_chunk_seconds: float = C.CHUNK_LENGTH,
) -> list[np.ndarray]:
    """Resample to the model rate and split into fixed-length chunks.

    Parameters
    ----------
    audio
        Waveform of shape [1, num_samples].
    audio_sample_rate
        Sample rate of ``audio``, in samples / second.
    model_sample_rate
        Sample rate required by the model. ``audio`` is resampled to this rate.
    model_chunk_seconds
        Split the audio into sequences of this many seconds. The final chunk may
        be shorter.

    Returns
    -------
    list[np.ndarray]
        List of waveform chunks, each of shape [1, chunk_samples].
    """
    if audio_sample_rate != model_sample_rate:
        audio = resampy.resample(audio, audio_sample_rate, model_sample_rate, axis=1)
        audio_sample_rate = model_sample_rate
    number_of_full_length_audio_chunks = int(
        audio.shape[1] // audio_sample_rate // model_chunk_seconds
    )
    last_sample_in_full_length_audio_chunks = int(
        audio_sample_rate * number_of_full_length_audio_chunks * model_chunk_seconds
    )
    if number_of_full_length_audio_chunks == 0:
        return [audio]

    return [
        *np.array_split(
            audio[:, :last_sample_in_full_length_audio_chunks],
            number_of_full_length_audio_chunks,
            axis=1,
        ),
    ]


def _periodic_hann(window_length: int) -> np.ndarray:
    """Return a periodic Hann window matching ``torch.hann_window(periodic=True)``.

    Parameters
    ----------
    window_length
        Number of points in the window.

    Returns
    -------
    np.ndarray
        The periodic Hann window, shape [window_length].
    """
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(window_length) / window_length)


def _mel_filterbank(
    n_freqs: int,
    n_mels: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Build an HTK mel filterbank matching ``torchaudio.melscale_fbanks(norm=None)``.

    Parameters
    ----------
    n_freqs
        Number of frequency bins (``n_fft // 2 + 1``).
    n_mels
        Number of mel bands.
    sample_rate
        Audio sample rate, in samples / second.
    f_min
        Lowest mel-band edge, in Hz.
    f_max
        Highest mel-band edge, in Hz.

    Returns
    -------
    np.ndarray
        Filterbank matrix of shape [n_freqs, n_mels] to post-multiply a
        magnitude spectrogram.
    """

    def hz_to_mel(freq: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    all_freqs = np.linspace(0.0, sample_rate // 2, n_freqs)
    m_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    f_diff = np.diff(f_pts)
    slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]
    down_slopes = -slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down_slopes, up_slopes)).astype(np.float32)


def _stft_magnitude(signal: np.ndarray) -> np.ndarray:
    """Compute a centered STFT magnitude spectrogram (matches ``torch.stft``).

    Parameters
    ----------
    signal
        1-D waveform, float.

    Returns
    -------
    np.ndarray
        Magnitude spectrogram of shape [num_frames, n_fft // 2 + 1].
    """
    win_length = round(C.SAMPLE_RATE * C.STFT_WINDOW_SECONDS)
    hop_length = round(C.SAMPLE_RATE * C.STFT_HOP_SECONDS)
    n_fft = C.N_FFT

    # center=True reflect padding, as used by torchaudio's spectrogram.
    pad = n_fft // 2
    padded = np.pad(signal, (pad, pad), mode="reflect")

    # Window of win_length centered inside the n_fft analysis buffer.
    window = np.zeros(n_fft, dtype=np.float64)
    offset = (n_fft - win_length) // 2
    window[offset : offset + win_length] = _periodic_hann(win_length)

    num_frames = 1 + (len(padded) - n_fft) // hop_length
    frames = np.empty((num_frames, n_fft), dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = padded[start : start + n_fft] * window

    # torchaudio uses power=2.0; the VGGish transform then takes the sqrt, which
    # is just the magnitude. Compute the magnitude directly.
    return np.abs(np.fft.rfft(frames, n_fft))


def wav_to_logmel_patches(segment: np.ndarray) -> np.ndarray:
    """Convert a 16 kHz mono waveform chunk to log-mel model patches.

    Parameters
    ----------
    segment
        Waveform of shape [1, num_samples] (or 1-D), float32 at ``SAMPLE_RATE``.

    Returns
    -------
    np.ndarray
        Patches of shape [num_patches, 1, 96, 64], float32. Empty (num_patches
        == 0) if the segment is shorter than one patch window.
    """
    data = np.asarray(segment, dtype=np.float64).reshape(-1)
    magnitude = _stft_magnitude(data)
    fb = _mel_filterbank(
        magnitude.shape[1], C.N_MELS, C.SAMPLE_RATE, C.MEL_MIN_HZ, C.MEL_MAX_HZ
    )
    log_mel = np.log(magnitude.astype(np.float32) @ fb + np.float32(C.LOG_OFFSET))

    window_frames = round(C.PATCH_WINDOW_SECONDS / C.STFT_HOP_SECONDS)
    hop_frames = round(C.PATCH_HOP_SECONDS / C.STFT_HOP_SECONDS)
    if log_mel.shape[0] < window_frames:
        return np.zeros((0, 1, window_frames, C.N_MELS), dtype=np.float32)

    num_patches = (log_mel.shape[0] - window_frames) // hop_frames + 1
    patches = np.stack(
        [
            log_mel[i * hop_frames : i * hop_frames + window_frames]
            for i in range(num_patches)
        ]
    )
    return patches[:, np.newaxis, :, :].astype(np.float32)
