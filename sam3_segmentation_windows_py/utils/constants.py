# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Model options
BACKBONE_SIZE = 1008
CONTEXT_LENGTH = 32
MASK_THRESHOLD = 0.0

# CLIP tokenizer padding.  SAM3's exported head derives its padding mask from
# ``tokenized == 0`` (matching sam3's own SimpleTokenizer, which zero-pads),
# so pad with id 0 here too -- NOT the CLIP end-of-text token -- or the model
# attends over the pad positions as real prompt content.
CLIP_PAD_TOKEN = "!"
CLIP_PAD_TOKEN_ID = 0

OVERLAY_COLORS = [
    (30, 144, 255),
    (255, 80, 80),
    (80, 220, 80),
    (255, 200, 40),
    (200, 40, 255),
    (40, 220, 200),
    (255, 140, 40),
    (40, 140, 255),
]
