# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import os

import numpy as np
import onnxruntime
from PIL import Image
from qai_hub_models.models._shared.stable_diffusion.app import StableDiffusionApp
from qai_hub_models.models._shared.stable_diffusion.model import make_scheduler
from qai_hub_models.utils.args import add_output_dir_arg
from qai_hub_models.utils.display import display_or_save_image, to_uint8
from qai_hub_models.utils.onnx.torch_wrapper import OnnxModelTorchWrapper
from transformers import CLIPTokenizer

if os.environ.get("ORT_LOG_LEVEL"):
    onnxruntime.set_default_logger_severity(int(os.environ["ORT_LOG_LEVEL"]))

DEFAULT_PROMPT = "A girl taking a walk at sunset"
HF_REPO = "sd2-community/stable-diffusion-2-1"


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="error",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt for stable diffusion",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=47,
        help="Random generator seed",
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        default=r"model\text_encoder.onnx",
        help="Text Encoder ONNX model path",
    )
    parser.add_argument(
        "--unet",
        type=str,
        default=r"model\unet.onnx",
        help="UNET ONNX model path",
    )
    parser.add_argument(
        "--vae-decoder",
        type=str,
        default=r"model\vae.onnx",
        help="VAE Decoder ONNX model path",
    )
    add_output_dir_arg(parser)
    args = parser.parse_args()

    # Load model
    print("Loading model and app...")
    sdapp = StableDiffusionApp(
        OnnxModelTorchWrapper.OnNPU(args.text_encoder),
        OnnxModelTorchWrapper.OnNPU(args.vae_decoder),
        OnnxModelTorchWrapper.OnNPU(args.unet),
        CLIPTokenizer.from_pretrained(HF_REPO, subfolder="tokenizer"),
        make_scheduler(HF_REPO, subfolder="scheduler"),
        channel_last_latent=True,
    )

    # Generate image
    print("Generating image...")
    image = sdapp.generate_image(args.prompt, args.num_steps, args.seed)
    pil_img = Image.fromarray(to_uint8(np.asarray(image))[0])
    display_or_save_image(pil_img, args.output_dir, filename="output.png")


if __name__ == "__main__":
    main()
