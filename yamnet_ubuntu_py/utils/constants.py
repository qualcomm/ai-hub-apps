# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Sample rate (Hz) YamNet expects. Input audio is resampled to this rate.
SAMPLE_RATE = 16000

# Split the input waveform into segments of this many seconds before feature
# extraction.
CHUNK_LENGTH = 0.98

# Short-time Fourier transform parameters for the log-mel front end.
STFT_WINDOW_SECONDS = 0.025  # 400 samples @ 16 kHz
STFT_HOP_SECONDS = 0.010  # 160 samples @ 16 kHz
N_FFT = 512  # next power of two above the 400-sample window

# Mel filterbank parameters.
N_MELS = 64
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = 7500.0
LOG_OFFSET = 0.001  # stabilizes log(mel) near zero

# Each model input patch spans this many 10 ms frames (0.96 s), and successive
# patches advance by PATCH_HOP_SECONDS.
MELS_AUDIO_LEN = 96
PATCH_WINDOW_SECONDS = 0.96
PATCH_HOP_SECONDS = 1.0

# Number of AudioSet classes the model scores.
NUM_CLASSES = 521

LABELS_FILENAME = "labels.txt"
