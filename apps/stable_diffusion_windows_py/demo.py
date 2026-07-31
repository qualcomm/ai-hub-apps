# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""Stable Diffusion v2.1 text-to-image on the Snapdragon NPU.

Classifier-free-guidance diffusion loop over three precompiled QNN ONNX graphs
(text_encoder, unet, vae), run via ONNX Runtime's QNN Execution Provider, with a
real CLIP tokenizer and a pure-numpy Euler scheduler (see utils/scheduler.py).

Usage:
  python demo.py --prompt "A girl taking a walk at sunset" --num-steps 20
"""

import argparse
import os

import numpy as np
import onnxruntime
from PIL import Image
from transformers import CLIPTokenizer
from utils.image_display import display_or_save_image, to_uint8
from utils.qnn_model import QuantizedModel, load_quantized_model, run_quantized
from utils.scheduler import EulerScheduler

if os.environ.get("ORT_LOG_LEVEL"):
    onnxruntime.set_default_logger_severity(int(os.environ["ORT_LOG_LEVEL"]))

DEFAULT_PROMPT = "A girl taking a walk at sunset"
HF_REPO = "sd2-community/stable-diffusion-2-1"
OUT_H, OUT_W = 512, 512


def encode_text_prompt(
    text_encoder: QuantizedModel, tokenizer: CLIPTokenizer, prompt: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the conditional and unconditional (empty-prompt) text embeddings."""
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="np",
    )
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""],
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )

    print(f"\nExtracting embeddings (inference on TextEncoder)\n{'-' * 50}")
    cond_embeddings = run_quantized(text_encoder, text_input.input_ids.astype(np.int32))
    uncond_embeddings = run_quantized(
        text_encoder, uncond_input.input_ids.astype(np.int32)
    )
    return cond_embeddings, uncond_embeddings


def run_diffusion_steps(
    unet: QuantizedModel,
    cond_embeddings: np.ndarray,
    uncond_embeddings: np.ndarray,
    num_steps: int,
    seed: int,
    guidance_scale: float,
) -> np.ndarray:
    """Run the classifier-free-guidance denoising loop, returning the final latent.

    The UNet expects/produces channel-last latents (1, 64, 64, 4); latents are
    kept channel-first (1, 4, 64, 64) between steps and transposed at the boundary.
    """
    scheduler = EulerScheduler(num_steps)

    latents_shape = (1, 4, OUT_H // 8, OUT_W // 8)
    latents = (
        np.random.default_rng(seed).standard_normal(latents_shape).astype(np.float32)
    )
    latents *= scheduler.init_noise_sigma

    for i, t in enumerate(scheduler.timesteps):
        print(f"\nStep: {i + 1}\n{'-' * 10}")
        time_input = np.array([[t]], dtype=np.float32)

        latent_input = scheduler.scale_model_input(latents, i).astype(np.float32)
        latent_input = np.transpose(latent_input, (0, 2, 3, 1))

        noise_cond = run_quantized(unet, latent_input, time_input, cond_embeddings)
        noise_uncond = run_quantized(unet, latent_input, time_input, uncond_embeddings)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        noise_pred = np.transpose(noise_pred, (0, 3, 1, 2))
        latents = scheduler.step(noise_pred, i, latents)

    return latents


def generate_image(
    text_encoder: QuantizedModel,
    unet: QuantizedModel,
    vae_decoder: QuantizedModel,
    tokenizer: CLIPTokenizer,
    prompt: str,
    num_steps: int,
    seed: int,
    guidance_scale: float = 7.5,
) -> np.ndarray:
    """Generate an image from ``prompt``, returned as RGB in [0, 1], shape (1, H, W, 3)."""
    cond_embeddings, uncond_embeddings = encode_text_prompt(
        text_encoder, tokenizer, prompt
    )
    latents = run_diffusion_steps(
        unet, cond_embeddings, uncond_embeddings, num_steps, seed, guidance_scale
    )
    latents = np.transpose(latents, (0, 2, 3, 1))  # channel-last for the VAE
    return run_quantized(vae_decoder, latents)


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
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="If specified, saves demo output (e.g. image) to this directory instead of displaying.",
    )
    args = parser.parse_args()

    print("Loading models and tokenizer...")
    text_encoder = load_quantized_model(args.text_encoder)
    unet = load_quantized_model(args.unet)
    vae_decoder = load_quantized_model(args.vae_decoder)
    tokenizer = CLIPTokenizer.from_pretrained(HF_REPO, subfolder="tokenizer")

    print("Generating image...")
    image = generate_image(
        text_encoder,
        unet,
        vae_decoder,
        tokenizer,
        args.prompt,
        args.num_steps,
        args.seed,
    )
    pil_img = Image.fromarray(to_uint8(image)[0])
    display_or_save_image(pil_img, args.output_dir, filename="output.png")


if __name__ == "__main__":
    main()
