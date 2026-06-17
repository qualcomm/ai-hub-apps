# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Measure Stable Diffusion QNN sensitivity to classifier-free guidance scale.

This script leaves demo.py unchanged. It reuses the v0.48 Snapdragon X Elite
QNN path and writes diagnostics under outputs/intermediate_debug by default.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import torch
from PIL import Image
from qai_hub_models.models._shared.stable_diffusion.app import OUT_H, OUT_W
from qai_hub_models.models._shared.stable_diffusion.model import make_scheduler
from qai_hub_models.utils.display import to_uint8
from qai_hub_models.utils.onnx.torch_wrapper import OnnxModelTorchWrapper
from transformers import CLIPTokenizer

from diagnose_intermediate_qnn import (
    DEFAULT_PROMPT,
    HF_REPO,
    MODEL_DIR,
    append_stat,
    as_numpy,
    channel_first,
    channel_last,
    ensure_dir,
    fmt,
    tensor_stats,
)


DEFAULT_GUIDANCE_SCALES = [0.0, 1.0, 3.0, 7.5, 15.0, 30.0]


def make_npu_models() -> tuple[OnnxModelTorchWrapper, OnnxModelTorchWrapper, OnnxModelTorchWrapper]:
    return (
        OnnxModelTorchWrapper.OnNPU(MODEL_DIR / "text_encoder.onnx"),
        OnnxModelTorchWrapper.OnNPU(MODEL_DIR / "vae.onnx"),
        OnnxModelTorchWrapper.OnNPU(MODEL_DIR / "unet.onnx"),
    )


def stat_value(stats: list[dict[str, Any]], name: str, key: str) -> Any:
    for stat in stats:
        if stat["name"] == name:
            return stat.get(key)
    return None


def image_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16))


def scale_tag(scale: float) -> str:
    return str(scale).replace(".", "p").replace("-", "m")


def run_one_scale(
    guidance_scale: float,
    scheduler: Any,
    unet: OnnxModelTorchWrapper,
    vae_decoder: OnnxModelTorchWrapper,
    cond_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    initial_latents: torch.Tensor,
    stats: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    tag = scale_tag(guidance_scale)
    latents = initial_latents.clone()
    first_noise_pred: torch.Tensor | None = None
    first_guidance_delta: torch.Tensor | None = None

    for index, timestep in enumerate(scheduler.timesteps):
        step = index + 1
        time_input = torch.as_tensor([[timestep]], dtype=torch.float32)
        latent_input = scheduler.scale_model_input(latents, timestep)
        latent_input_nhwc = channel_last(latent_input)

        noise_cond = unet(latent_input_nhwc, time_input, cond_embeddings)
        noise_uncond = unet(latent_input_nhwc, time_input, uncond_embeddings)
        noise_cond_nchw = channel_first(noise_cond)
        noise_uncond_nchw = channel_first(noise_uncond)
        guidance_delta = noise_cond_nchw - noise_uncond_nchw
        noise_pred = noise_uncond_nchw + guidance_scale * guidance_delta

        append_stat(stats, f"guidance_{tag}.step_{step}.cond_uncond_delta", guidance_delta)
        append_stat(stats, f"guidance_{tag}.step_{step}.noise_pred", noise_pred)

        if first_noise_pred is None:
            first_noise_pred = noise_pred.clone()
            first_guidance_delta = guidance_delta.clone()

        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
        append_stat(stats, f"guidance_{tag}.step_{step}.latent_after_scheduler", latents)

    vae_latent_nhwc = channel_last(latents)
    image_float = vae_decoder(vae_latent_nhwc)
    image_uint8 = to_uint8(as_numpy(image_float))[0]
    Image.fromarray(image_uint8).save(output_dir / f"guidance_{tag}.png")

    append_stat(stats, f"guidance_{tag}.final_latent", latents)
    append_stat(stats, f"guidance_{tag}.image_float", image_float)
    append_stat(stats, f"guidance_{tag}.image_uint8", image_uint8)

    assert first_noise_pred is not None
    assert first_guidance_delta is not None
    return {
        "scale": guidance_scale,
        "tag": tag,
        "first_noise_pred": first_noise_pred,
        "first_guidance_delta": first_guidance_delta,
        "final_latent": latents,
        "image_uint8": image_uint8,
    }


def write_report(
    path: Path,
    env: dict[str, Any],
    stats: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    conclusion: str,
) -> None:
    lines = [
        "# Guidance Sensitivity Debug",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(env, indent=2),
        "```",
        "",
        "## Scale Comparison",
        "",
        "| guidance_scale | noise_pred_std | cond_uncond_delta_std | final_latent_std | image_std | noise_pred_absdiff_vs_0_mean | final_latent_absdiff_vs_0_mean | image_absdiff_vs_0_mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {scale} | {noise_pred_std} | {delta_std} | {latent_std} | {image_std} | {noise_diff_mean} | {latent_diff_mean} | {image_diff_mean} |".format(
                scale=row["scale"],
                noise_pred_std=fmt(row["noise_pred_std"]),
                delta_std=fmt(row["cond_uncond_delta_std"]),
                latent_std=fmt(row["final_latent_std"]),
                image_std=fmt(row["image_std"]),
                noise_diff_mean=fmt(row["noise_pred_absdiff_vs_0_mean"]),
                latent_diff_mean=fmt(row["final_latent_absdiff_vs_0_mean"]),
                image_diff_mean=fmt(row["image_absdiff_vs_0_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
            "## Output Images",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `guidance_{row['tag']}.png`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--guidance-scales",
        nargs="+",
        type=float,
        default=DEFAULT_GUIDANCE_SCALES,
    )
    parser.add_argument("--output-dir", default="outputs/intermediate_debug")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    stats: list[dict[str, Any]] = []

    print("Loading tokenizer, scheduler, and QNN models...")
    tokenizer = CLIPTokenizer.from_pretrained(HF_REPO, subfolder="tokenizer")
    scheduler = make_scheduler(HF_REPO, subfolder="scheduler")
    text_encoder, vae_decoder, unet = make_npu_models()

    text_input = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_input = tokenizer(
        [""],
        padding="max_length",
        max_length=text_input.input_ids.shape[-1],
        return_tensors="pt",
    )
    tokens = text_input.input_ids.type(torch.int32)
    uncond_tokens = uncond_input.input_ids.type(torch.int32)
    cond_embeddings = text_encoder(tokens)
    uncond_embeddings = text_encoder(uncond_tokens)
    append_stat(stats, "text_encoder.cond_minus_uncond", cond_embeddings - uncond_embeddings)

    scheduler.set_timesteps(args.num_steps)
    generator = torch.manual_seed(args.seed)
    initial_latents = torch.randn((1, 4, OUT_H // 8, OUT_W // 8), generator=generator)
    initial_latents = initial_latents * scheduler.init_noise_sigma
    append_stat(stats, "initial_latents", initial_latents)

    print("Running guidance sensitivity sweep...")
    results = []
    with torch.no_grad():
        for scale in args.guidance_scales:
            print(f"guidance_scale={scale}")
            results.append(
                run_one_scale(
                    scale,
                    scheduler,
                    unet,
                    vae_decoder,
                    cond_embeddings,
                    uncond_embeddings,
                    initial_latents,
                    stats,
                    output_dir,
                )
            )

    baseline = results[0]
    rows = []
    for result in results:
        tag = result["tag"]
        noise_diff = np.abs(
            as_numpy(result["first_noise_pred"]).astype(np.float32)
            - as_numpy(baseline["first_noise_pred"]).astype(np.float32)
        )
        latent_diff = np.abs(
            as_numpy(result["final_latent"]).astype(np.float32)
            - as_numpy(baseline["final_latent"]).astype(np.float32)
        )
        img_diff = image_abs_diff(result["image_uint8"], baseline["image_uint8"])
        append_stat(stats, f"guidance_{tag}.noise_pred_absdiff_vs_0", noise_diff)
        append_stat(stats, f"guidance_{tag}.final_latent_absdiff_vs_0", latent_diff)
        append_stat(stats, f"guidance_{tag}.image_absdiff_vs_0", img_diff)
        rows.append(
            {
                "scale": result["scale"],
                "tag": tag,
                "noise_pred_std": stat_value(stats, f"guidance_{tag}.step_1.noise_pred", "std"),
                "cond_uncond_delta_std": stat_value(
                    stats, f"guidance_{tag}.step_1.cond_uncond_delta", "std"
                ),
                "final_latent_std": stat_value(stats, f"guidance_{tag}.final_latent", "std"),
                "image_std": stat_value(stats, f"guidance_{tag}.image_uint8", "std"),
                "noise_pred_absdiff_vs_0_mean": float(noise_diff.mean()),
                "final_latent_absdiff_vs_0_mean": float(latent_diff.mean()),
                "image_absdiff_vs_0_mean": float(img_diff.mean()),
            }
        )

    max_image_diff = max(row["image_absdiff_vs_0_mean"] for row in rows)
    max_latent_diff = max(row["final_latent_absdiff_vs_0_mean"] for row in rows)
    max_noise_diff = max(row["noise_pred_absdiff_vs_0_mean"] for row in rows)
    conclusion = (
        "Guidance scale changes do affect noise_pred, final latent, and image output. "
        "This argues against a totally disconnected guidance path. "
        "If the generated image still looks low-information, the remaining suspicion is weak UNet text-conditioning sensitivity inside the QNN context rather than Python-side guidance_scale being ignored."
    )
    if max_image_diff < 2 or max_latent_diff < 1e-3 or max_noise_diff < 1e-3:
        conclusion = (
            "Guidance scale changes produced only small output differences. Treat this as evidence for a UNet text-conditioning path or context.bin sensitivity problem."
        )

    env = {
        "python": sys.version,
        "platform_machine": platform.machine(),
        "onnxruntime_version": onnxruntime.__version__,
        "onnxruntime_providers": onnxruntime.get_available_providers(),
        "prompt": args.prompt,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "guidance_scales": args.guidance_scales,
    }
    (output_dir / "guidance_sensitivity.json").write_text(
        json.dumps({"environment": env, "stats": stats, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir / "guidance_sensitivity.md", env, stats, rows, conclusion)


if __name__ == "__main__":
    main()
