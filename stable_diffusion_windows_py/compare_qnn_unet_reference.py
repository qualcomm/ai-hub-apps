# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Compare QNN UNet text-conditioning sensitivity with a PyTorch UNet baseline.

This script leaves demo.py unchanged. It uses the same prompt, seed, latent,
timestep, and QNN TextEncoder embeddings for both UNet paths, then compares the
conditional/unconditional noise delta.
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
from diffusers import UNet2DConditionModel
from qai_hub_models.models._shared.stable_diffusion.app import OUT_H, OUT_W
from qai_hub_models.models._shared.stable_diffusion.model import make_scheduler
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
)


def tensor_abs_diff(a: Any, b: Any) -> np.ndarray:
    return np.abs(as_numpy(a).astype(np.float32) - as_numpy(b).astype(np.float32))


def find_stat(stats: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((item for item in stats if item["name"] == name), None)


def stat_std(stats: list[dict[str, Any]], name: str) -> float | None:
    stat = find_stat(stats, name)
    return None if stat is None else stat["std"]


def load_qnn_models() -> tuple[OnnxModelTorchWrapper, OnnxModelTorchWrapper]:
    return (
        OnnxModelTorchWrapper.OnNPU(MODEL_DIR / "text_encoder.onnx"),
        OnnxModelTorchWrapper.OnNPU(MODEL_DIR / "unet.onnx"),
    )


def run_compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    stats: list[dict[str, Any]] = []
    notes: list[str] = []

    print("Loading tokenizer, scheduler, QNN TextEncoder, and QNN UNet...")
    tokenizer = CLIPTokenizer.from_pretrained(HF_REPO, subfolder="tokenizer")
    scheduler = make_scheduler(HF_REPO, subfolder="scheduler")
    qnn_text_encoder, qnn_unet = load_qnn_models()

    print("Loading PyTorch UNet baseline...")
    try:
        torch_unet = UNet2DConditionModel.from_pretrained(
            HF_REPO,
            subfolder="unet",
            local_files_only=args.local_files_only,
        )
    except Exception as exc:  # noqa: BLE001 - report the blocker.
        env = environment(args)
        notes.append(f"PyTorch UNet baseline load failed: {type(exc).__name__}: {exc}")
        write_outputs(output_dir, env, stats, notes)
        return

    torch_unet.eval()
    torch_unet.to("cpu")

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

    with torch.no_grad():
        cond_embeddings = qnn_text_encoder(tokens)
        uncond_embeddings = qnn_text_encoder(uncond_tokens)
        append_stat(stats, "text_emb.cond", cond_embeddings)
        append_stat(stats, "text_emb.uncond", uncond_embeddings)
        append_stat(stats, "text_emb.cond_minus_uncond", cond_embeddings - uncond_embeddings)

        scheduler.set_timesteps(args.num_steps)
        timestep = scheduler.timesteps[args.timestep_index]
        generator = torch.manual_seed(args.seed)
        latents = torch.randn((1, 4, OUT_H // 8, OUT_W // 8), generator=generator)
        latents = latents * scheduler.init_noise_sigma
        latent_input = scheduler.scale_model_input(latents, timestep)
        time_input_qnn = torch.as_tensor([[timestep]], dtype=torch.float32)
        time_input_torch = torch.as_tensor([timestep], dtype=torch.float32)

        append_stat(stats, "latent.initial", latents)
        append_stat(stats, "unet.input.latent_nchw", latent_input)
        append_stat(stats, "unet.input.timestep_qnn", time_input_qnn)
        append_stat(stats, "unet.input.timestep_torch", time_input_torch)

        print("Running QNN UNet cond/uncond...")
        latent_input_nhwc = channel_last(latent_input)
        qnn_noise_cond = channel_first(qnn_unet(latent_input_nhwc, time_input_qnn, cond_embeddings))
        qnn_noise_uncond = channel_first(
            qnn_unet(latent_input_nhwc, time_input_qnn, uncond_embeddings)
        )
        qnn_delta = qnn_noise_cond - qnn_noise_uncond
        append_stat(stats, "qnn.noise_cond", qnn_noise_cond)
        append_stat(stats, "qnn.noise_uncond", qnn_noise_uncond)
        append_stat(stats, "qnn.noise_cond_minus_uncond", qnn_delta)

        print("Running PyTorch UNet cond/uncond on CPU...")
        ref_noise_cond = torch_unet(
            latent_input,
            time_input_torch,
            encoder_hidden_states=cond_embeddings,
        ).sample
        ref_noise_uncond = torch_unet(
            latent_input,
            time_input_torch,
            encoder_hidden_states=uncond_embeddings,
        ).sample
        ref_delta = ref_noise_cond - ref_noise_uncond
        append_stat(stats, "reference.noise_cond", ref_noise_cond)
        append_stat(stats, "reference.noise_uncond", ref_noise_uncond)
        append_stat(stats, "reference.noise_cond_minus_uncond", ref_delta)

        append_stat(stats, "compare.noise_cond_absdiff", tensor_abs_diff(qnn_noise_cond, ref_noise_cond))
        append_stat(
            stats,
            "compare.noise_uncond_absdiff",
            tensor_abs_diff(qnn_noise_uncond, ref_noise_uncond),
        )
        append_stat(stats, "compare.delta_absdiff", tensor_abs_diff(qnn_delta, ref_delta))

    env = environment(args)
    env["timestep"] = float(as_numpy(time_input_torch)[0])
    env["reference_unet"] = HF_REPO
    write_outputs(output_dir, env, stats, notes)


def environment(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform_machine": platform.machine(),
        "onnxruntime_version": onnxruntime.__version__,
        "onnxruntime_providers": onnxruntime.get_available_providers(),
        "prompt": args.prompt,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "timestep_index": args.timestep_index,
        "local_files_only": args.local_files_only,
    }


def write_outputs(
    output_dir: Path,
    env: dict[str, Any],
    stats: list[dict[str, Any]],
    notes: list[str],
) -> None:
    (output_dir / "unet_reference_compare.json").write_text(
        json.dumps({"environment": env, "stats": stats, "notes": notes}, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir / "unet_reference_compare.md", env, stats, notes)


def write_report(
    path: Path,
    env: dict[str, Any],
    stats: list[dict[str, Any]],
    notes: list[str],
) -> None:
    key_names = [
        "text_emb.cond_minus_uncond",
        "qnn.noise_cond",
        "qnn.noise_uncond",
        "qnn.noise_cond_minus_uncond",
        "reference.noise_cond",
        "reference.noise_uncond",
        "reference.noise_cond_minus_uncond",
        "compare.delta_absdiff",
    ]
    qnn_delta_std = stat_std(stats, "qnn.noise_cond_minus_uncond")
    ref_delta_std = stat_std(stats, "reference.noise_cond_minus_uncond")
    ratio = None
    if qnn_delta_std is not None and ref_delta_std:
        ratio = qnn_delta_std / ref_delta_std

    if notes:
        conclusion = (
            "Reference UNet comparison was not completed. See notes for the blocker and retry after the baseline UNet is available."
        )
    elif ratio is not None and ratio < 0.1:
        conclusion = (
            f"QNN UNet text-conditioning delta is much smaller than the PyTorch baseline (std ratio {ratio:.6g}). Treat `unet_qairt_context.bin` or UNet conversion settings as the primary suspect."
        )
    elif ratio is not None:
        conclusion = (
            f"QNN UNet text-conditioning delta is not extremely smaller than the PyTorch baseline (std ratio {ratio:.6g}). The low-information output may require checking scheduler/VAE/full denoising behavior next."
        )
    else:
        conclusion = "Could not compute a QNN/reference delta ratio."

    lines = [
        "# QNN UNet vs PyTorch UNet Reference Compare",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(env, indent=2),
        "```",
        "",
        "## Key Statistics",
        "",
        "| tensor | shape | dtype | min | max | mean | std | zero_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in key_names:
        stat = find_stat(stats, name)
        if stat is None:
            continue
        lines.append(
            f"| {name} | {stat['shape']} | {stat['dtype']} | {fmt(stat['min'])} | {fmt(stat['max'])} | {fmt(stat['mean'])} | {fmt(stat['std'])} | {fmt(stat['zero_ratio'])} |"
        )

    lines.extend(["", "## Delta Ratio", ""])
    lines.append(f"- QNN delta std: `{fmt(qnn_delta_std)}`")
    lines.append(f"- PyTorch reference delta std: `{fmt(ref_delta_std)}`")
    lines.append(f"- QNN / reference std ratio: `{fmt(ratio)}`")
    lines.extend(["", "## Conclusion", "", conclusion])

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--timestep-index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/intermediate_debug")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_compare(parse_args())
