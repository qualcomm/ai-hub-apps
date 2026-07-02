# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Compare two images using mean absolute difference.

Usage:
    python compare_images.py <generated> <reference> [--threshold 20]

Exits 0 if MAD < threshold, 1 otherwise.
"""

import argparse
import sys

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", help="Path to generated image")
    parser.add_argument("reference", help="Path to reference image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Maximum allowed mean absolute difference (0-255 scale)",
    )
    args = parser.parse_args()

    gen = np.array(Image.open(args.generated).convert("RGB"), dtype=float)
    ref = np.array(Image.open(args.reference).convert("RGB"), dtype=float)

    if gen.shape != ref.shape:
        ref = np.array(
            Image.open(args.reference)
            .convert("RGB")
            .resize((gen.shape[1], gen.shape[0]), Image.LANCZOS),
            dtype=float,
        )

    mad = float(np.mean(np.abs(gen - ref)))
    print(f"Mean absolute difference: {mad:.4f} (threshold: {args.threshold})")

    if mad < args.threshold:
        print("PASS: images match.")
        sys.exit(0)
    else:
        print("FAIL: images differ.")
        sys.exit(1)


if __name__ == "__main__":
    main()
