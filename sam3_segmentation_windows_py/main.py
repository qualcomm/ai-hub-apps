# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""SAM3 (Segment Anything Model 3) text-prompted segmentation on the Snapdragon NPU.

Two-stage pipeline via ONNX Runtime QNN Execution Provider:
  vision_backbone.onnx  image[1,3,1008,1008]
                        -> backbone_fpn_0/1/2 FPN features
  head.onnx             tokenized[1,32] + FPN features
                        -> pred_boxes, scores, pred_masks

The vision backbone and head are precompiled QNN context binaries exported by
AI Hub; onnxruntime-qnn loads and runs them directly on the Snapdragon HTP.

Usage:
  python main.py --image photo.jpg --text-prompts "cup"
  python main.py --backbone models\\vision_backbone.onnx ^
                 --head     models\\head.onnx ^
                 --image    photo.jpg --text-prompts "cup"
"""

import argparse
import os
import platform
import sys
from pathlib import Path

if platform.machine() != "ARM64":
    raise RuntimeError(
        f"ARM64 machine (Python ARM64) is required for running the app but detected {platform.machine()}. Please uninstall and install ARM64 Python and rerun the app. Tip: winget uninstall python; <app_source>/install_runtime.ps1"
    )
import numpy as np
import utils.constants as C
from qai_hub_apps_utils.bbox_processing import box_xywh_to_xyxy, nms
from qai_hub_apps_utils.onnxruntime_qnn import open_qnn_session
from tokenizers import Tokenizer
from utils.draw import draw_predictions
from utils.input_processing import preprocess_image


def load_tokenizer(tokenizer_arg: str) -> Tokenizer:
    """Load the CLIP tokenizer, configured for fixed-length padding/truncation.

    Parameters
    ----------
    tokenizer_arg
        Either a path to a local ``tokenizer.json`` file (or a directory
        containing one) or a HuggingFace model name to download. The CLIP
        tokenizer is a public model and needs no HuggingFace token.

    Returns
    -------
    Tokenizer
        Tokenizer that pads/truncates every prompt to ``C.CONTEXT_LENGTH``.
    """
    path = Path(tokenizer_arg)
    if path.is_dir():
        path = path / "tokenizer.json"
    if path.is_file():
        tok = Tokenizer.from_file(str(path))
    else:
        # HuggingFace model name — downloads the public CLIP tokenizer (no token).
        tok = Tokenizer.from_pretrained(tokenizer_arg)

    tok.enable_truncation(max_length=C.CONTEXT_LENGTH)
    tok.enable_padding(
        length=C.CONTEXT_LENGTH,
        pad_id=C.CLIP_PAD_TOKEN_ID,
        pad_token=C.CLIP_PAD_TOKEN,
    )
    return tok


def main(args: argparse.Namespace) -> None:
    for path, name in ((args.backbone, "backbone"), (args.head, "head")):
        if not os.path.exists(path):
            sys.exit(
                f"ERROR: SAM3 {name} ONNX not found: {path}\n"
                "SAM3 is export-only: fetch the app together with an exported model "
                "bundle via the CLI, e.g.\n"
                "  qai-hub-apps fetch sam3_segmentation_windows_py "
                "--model <path/to/exported_bundle>\n"
                "which places vision_backbone.onnx + head.onnx under models/."
            )

    text_prompts = args.text_prompts
    if text_prompts is None:
        # Ask before the slow model loads rather than after them.
        try:
            text_prompts = input(
                "Enter what to segment (comma-separated, e.g. cup,person): "
            )
        except EOFError:
            text_prompts = ""

    prompts = [p.strip() for p in text_prompts.split(",") if p.strip()]
    if not prompts:
        sys.exit('ERROR: no text prompt provided. Pass one with --text-prompts "cup".')

    print(f"image  : {args.image}")
    print(f"prompts: {prompts}")

    print("loading tokenizer…")
    tok = load_tokenizer(args.tokenizer)
    encodings = tok.encode_batch(prompts)
    token_ids = np.array([e.ids for e in encodings], dtype=np.int32)

    image, orig_w, orig_h, x = preprocess_image(args.image)

    print("loading vision backbone…")
    bb = open_qnn_session(args.backbone)
    f0, f1, f2 = bb.run(
        ["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"],
        {"image": x},
    )

    print("loading head…")
    hd = open_qnn_session(args.head)

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_labels: list[str] = []

    for i, prompt in enumerate(prompts):
        pb, sc, pm = hd.run(
            ["pred_boxes", "scores", "pred_masks"],
            {
                "tokenized": token_ids[i : i + 1],
                "backbone_fpn_0": f0,
                "backbone_fpn_1": f1,
                "backbone_fpn_2": f2,
            },
        )
        s = sc[0]
        print(f"  '{prompt}': scores [{s.min():.3f}..{s.max():.3f}]")
        # box_xywh_to_xyxy returns normalized (x1, y1, x2, y2); scale to pixels.
        boxes_xyxy = box_xywh_to_xyxy(pb[0], flat_boxes=True)
        boxes_xyxy[:, [0, 2]] *= orig_w
        boxes_xyxy[:, [1, 3]] *= orig_h
        all_boxes.append(boxes_xyxy)
        all_scores.append(s)
        all_masks.append(pm[0])
        all_labels.extend([prompt] * len(s))

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    masks = np.concatenate(all_masks)

    keep = np.where(scores >= args.confidence)[0]
    boxes, scores, masks = boxes[keep], scores[keep], masks[keep]
    all_labels = [all_labels[i] for i in keep]

    if len(scores):
        kept = nms(boxes, scores, args.nms_iou)
        boxes, scores, masks = boxes[kept], scores[kept], masks[kept]
        all_labels = [all_labels[i] for i in kept]

    print(f"detections: {len(scores)}")
    out = (
        draw_predictions(image, boxes, masks, scores, all_labels)
        if len(scores)
        else image
    )
    if args.output is None:
        out.show()
    else:
        out.save(args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        default=os.path.join("models", "vision_backbone.onnx"),
        help="SAM3 vision backbone ONNX (default: models/vision_backbone.onnx)",
    )
    parser.add_argument(
        "--head",
        default=os.path.join("models", "head.onnx"),
        help="SAM3 head ONNX (default: models/head.onnx)",
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--text-prompts",
        default=None,
        help="Comma-separated text prompts, e.g. 'cup,person,bowl'. Omit to be "
        "prompted.",
    )
    parser.add_argument(
        "--tokenizer",
        default="openai/clip-vit-base-patch32",
        help=(
            "CLIP tokenizer: path to a local tokenizer.json (or a directory "
            "containing one) or a HuggingFace model name "
            "(default: openai/clip-vit-base-patch32)"
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Min confidence score after fp16 calibration (default: 0.5)",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output overlay image path. Omit to open the overlay in the "
        "default image viewer instead of writing it.",
    )

    args = parser.parse_args()
    main(args)
