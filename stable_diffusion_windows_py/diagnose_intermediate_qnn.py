# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Collect intermediate tensor statistics for the Stable Diffusion QNN demo.

This script intentionally leaves demo.py unchanged. It mirrors the v0.48
Snapdragon X Elite QNN path and writes diagnostics under
outputs/intermediate_debug by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import torch
from PIL import Image
from qai_hub_models.models._shared.stable_diffusion.app import OUT_H, OUT_W
from qai_hub_models.models._shared.stable_diffusion.model import make_scheduler
from qai_hub_models.utils.display import to_uint8
from qai_hub_models.utils.onnx.torch_wrapper import OnnxModelTorchWrapper
from transformers import CLIPTokenizer


HF_REPO = "sd2-community/stable-diffusion-2-1"
DEFAULT_PROMPT = "A girl taking a walk at sunset"
DEFAULT_ALT_PROMPT = "A red sports car parked in a snowy mountain at night"
MODEL_DIR = Path("model")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_stats(name: str, value: Any) -> dict[str, Any]:
    arr = as_numpy(value)
    flat = arr.reshape(-1) if arr.size else arr

    if arr.size == 0:
        return {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(arr.size),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "nan_count": 0,
            "inf_count": 0,
            "zero_count": 0,
            "zero_ratio": None,
        }

    if np.issubdtype(arr.dtype, np.number):
        numeric = flat.astype(np.float64, copy=False)
        finite = numeric[np.isfinite(numeric)]
        nan_count = int(np.isnan(numeric).sum())
        inf_count = int(np.isinf(numeric).sum())
        zero_count = int((numeric == 0).sum())
        return {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(arr.size),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std()) if finite.size else None,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_count": zero_count,
            "zero_ratio": float(zero_count / arr.size),
        }

    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "nan_count": None,
        "inf_count": None,
        "zero_count": None,
        "zero_ratio": None,
    }


def append_stat(stats: list[dict[str, Any]], name: str, value: Any) -> dict[str, Any]:
    stat = tensor_stats(name, value)
    stats.append(stat)
    return stat


def channel_last(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 3, 1).contiguous()


def channel_first(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 3, 1, 2).contiguous()


def onnx_type_to_str(type_proto: onnx.TypeProto) -> str:
    tensor_type = type_proto.tensor_type
    if not tensor_type.elem_type:
        return str(type_proto)
    return onnx.TensorProto.DataType.Name(tensor_type.elem_type)


def onnx_shape(value_info: onnx.ValueInfoProto) -> list[str | int]:
    dims: list[str | int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def collect_onnx_io(model_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in model_paths:
        model = onnx.load(path, load_external_data=False)
        initializer_names = {init.name for init in model.graph.initializer}
        inputs = [item for item in model.graph.input if item.name not in initializer_names]
        for direction, values in [("input", inputs), ("output", model.graph.output)]:
            for value in values:
                rows.append(
                    {
                        "model": path.name,
                        "direction": direction,
                        "name": value.name,
                        "shape": onnx_shape(value),
                        "dtype": onnx_type_to_str(value.type),
                    }
                )
    return rows


def write_onnx_io(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ONNX I/O",
        "",
        "| model | direction | name | shape | dtype |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['direction']} | {row['name']} | {row['shape']} | {row['dtype']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_model_files(model_dir: Path) -> list[dict[str, Any]]:
    expected = [
        "metadata.yaml",
        "text_encoder.onnx",
        "text_encoder_qairt_context.bin",
        "unet.onnx",
        "unet_qairt_context.bin",
        "vae.onnx",
        "vae_qairt_context.bin",
    ]
    rows = []
    for name in expected:
        path = model_dir / name
        rows.append(
            {
                "name": name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return rows


def write_model_files(path: Path, model_dir: Path, rows: list[dict[str, Any]]) -> None:
    metadata_path = model_dir / "metadata.yaml"
    metadata = metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else ""
    is_v048 = "onnx_runtime: 1.24.1" in metadata and "qairt: 2.42.0" in metadata

    lines = [
        "# Model Files",
        "",
        f"Model directory: `{model_dir}`",
        "",
        f"v0.48.0 metadata match: `{is_v048}`",
        "",
        "| file | exists | size_bytes |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['name']} | {row['exists']} | {row['size_bytes']} |")
    lines.extend(["", "## metadata.yaml", "", "```yaml", metadata.rstrip(), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_npu_models() -> tuple[OnnxModelTorchWrapper, OnnxModelTorchWrapper, OnnxModelTorchWrapper]:
    return (
        OnnxModelTorchWrapper.OnNPU(MODEL_DIR / "text_encoder.onnx"),
        OnnxModelTorchWrapper.OnNPU(MODEL_DIR / "vae.onnx"),
        OnnxModelTorchWrapper.OnNPU(MODEL_DIR / "unet.onnx"),
    )


def get_qdq_params(model: OnnxModelTorchWrapper, input_name: str) -> tuple[float, int, str]:
    input_details = model.inputs[input_name]
    if input_details.qdq_params is None:
        raise ValueError(f"{input_name} has no QDQ params")
    return (
        float(input_details.qdq_params.scale),
        int(input_details.qdq_params.zero_point),
        str(input_details.dtype),
    )


def quantize_like_wrapper(value: Any, scale: float, zero_point: int, dtype: str) -> np.ndarray:
    arr = as_numpy(value).astype(np.float64)
    quantized = np.rint(arr / scale) + zero_point
    info = np.iinfo(np.dtype(dtype))
    return quantized.clip(info.min, info.max).astype(np.dtype(dtype))


def dequantize_like_wrapper(value: Any, scale: float, zero_point: int) -> np.ndarray:
    return ((as_numpy(value).astype(np.int32) - zero_point) * np.float64(scale)).astype(np.float32)


def append_quantization_stats(
    stats: list[dict[str, Any]],
    prefix: str,
    before: Any,
    quantized: Any,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    dequantized = dequantize_like_wrapper(quantized, scale, zero_point)
    before_np = as_numpy(before).astype(np.float32)
    append_stat(stats, f"{prefix}.float32_before_quant", before_np)
    append_stat(stats, f"{prefix}.uint16_after_quant", quantized)
    append_stat(stats, f"{prefix}.dequantized_after_quant", dequantized)
    append_stat(stats, f"{prefix}.quantization_abs_error", np.abs(dequantized - before_np))
    return dequantized


def write_unet_text_emb_report(
    path: Path,
    rows: list[dict[str, Any]],
    wrapper_notes: list[str],
) -> None:
    lines = [
        "# UNet Text Embedding Input Debug",
        "",
        "## OnnxModelTorchWrapper._prepare_inputs()",
        "",
        "The wrapper prepares each input by converting tensors to numpy arrays, then quantizes float inputs when the ONNX input is an integer tensor with QDQ parameters and the input name is enabled in `quantize_user_input`.",
        "",
        "Quantization formula observed from the installed `qai_hub_models==0.48.0` wrapper:",
        "",
        "```text",
        "uint_value = round(float_value / scale) + zero_point",
        "uint_value = clip(uint_value, dtype_min, dtype_max).astype(dtype)",
        "```",
        "",
        "## Metadata / Wrapper QDQ",
        "",
        "| source | input | dtype | scale | zero_point |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['input']} | {row['dtype']} | {row['scale']} | {row['zero_point']} |"
        )
    lines.extend(["", "## Findings", ""])
    lines.extend(wrapper_notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_cpu_compare(
    output_dir: Path,
    tokens: torch.Tensor,
    vae_latent: torch.Tensor,
    npu_text_embedding: torch.Tensor,
    npu_vae_output: torch.Tensor,
) -> list[str]:
    lines: list[str] = ["# CPU Comparison", ""]
    for model_name, model_path, inputs, npu_output in [
        ("text_encoder", MODEL_DIR / "text_encoder.onnx", (tokens,), npu_text_embedding),
        ("vae", MODEL_DIR / "vae.onnx", (vae_latent,), npu_vae_output),
    ]:
        lines.append(f"## {model_name}")
        try:
            model = OnnxModelTorchWrapper.OnCPU(model_path)
            cpu_output = model(*inputs)
            cpu_np = as_numpy(cpu_output)
            npu_np = as_numpy(npu_output)
            diff = cpu_np.astype(np.float64) - npu_np.astype(np.float64)
            lines.append("")
            lines.append("CPU run: OK")
            lines.append(f"CPU stats: `{tensor_stats(model_name + '.cpu_output', cpu_output)}`")
            lines.append(f"NPU stats: `{tensor_stats(model_name + '.npu_output', npu_output)}`")
            lines.append(f"Abs diff stats: `{tensor_stats(model_name + '.abs_diff', np.abs(diff))}`")
        except Exception as exc:  # noqa: BLE001 - diagnostic report should capture all failures.
            lines.append("")
            lines.append("CPU run: FAILED")
            lines.append(f"Reason: `{type(exc).__name__}: {exc}`")
        lines.append("")
    (output_dir / "cpu_compare.md").write_text("\n".join(lines), encoding="utf-8")
    return lines


def run_diagnostics(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    stats: list[dict[str, Any]] = []
    notes: list[str] = []

    print("Loading tokenizer, scheduler, and QNN models...")
    tokenizer = CLIPTokenizer.from_pretrained(HF_REPO, subfolder="tokenizer")
    scheduler = make_scheduler(HF_REPO, subfolder="scheduler")
    text_encoder, vae_decoder, unet = make_npu_models()
    text_emb_scale, text_emb_zero_point, text_emb_dtype = get_qdq_params(unet, "text_emb")
    qdq_rows = [
        {
            "source": "metadata.yaml / OnnxModelTorchWrapper",
            "input": "unet.text_emb",
            "dtype": text_emb_dtype,
            "scale": text_emb_scale,
            "zero_point": text_emb_zero_point,
        }
    ]

    print("Encoding prompt...")
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
    alt_input = tokenizer(
        args.alt_prompt,
        padding="max_length",
        max_length=text_input.input_ids.shape[-1],
        return_tensors="pt",
    )

    tokens = text_input.input_ids.type(torch.int32)
    uncond_tokens = uncond_input.input_ids.type(torch.int32)
    alt_tokens = alt_input.input_ids.type(torch.int32)
    append_stat(stats, "text_encoder.input.tokens", tokens)
    append_stat(stats, "text_encoder.input.uncond_tokens", uncond_tokens)
    append_stat(stats, "text_encoder.input.alt_tokens", alt_tokens)
    cond_embeddings = text_encoder(tokens)
    uncond_embeddings = text_encoder(uncond_tokens)
    alt_embeddings = text_encoder(alt_tokens)
    append_stat(stats, "text_encoder.output.cond_embedding", cond_embeddings)
    append_stat(stats, "text_encoder.output.uncond_embedding", uncond_embeddings)
    append_stat(stats, "text_encoder.output.alt_embedding", alt_embeddings)
    append_stat(stats, "text_encoder.output.cond_minus_uncond", cond_embeddings - uncond_embeddings)
    append_stat(stats, "text_encoder.output.cond_minus_alt", cond_embeddings - alt_embeddings)

    print("Running diffusion diagnostics...")
    with torch.no_grad():
        scheduler.set_timesteps(args.num_steps)
        generator = torch.manual_seed(args.seed)
        latents = torch.randn((1, 4, OUT_H // 8, OUT_W // 8), generator=generator)
        append_stat(stats, "latent.initial.randn", latents)
        latents = latents * scheduler.init_noise_sigma
        append_stat(stats, "latent.initial.scaled", latents)

        first_timestep = scheduler.timesteps[0]
        first_time_input = torch.as_tensor([[first_timestep]], dtype=torch.float32)
        first_latent_input_nhwc = channel_last(scheduler.scale_model_input(latents, first_timestep))
        prompt_a_prepared = unet._prepare_inputs(
            (first_latent_input_nhwc, first_time_input, cond_embeddings)
        )
        prompt_b_prepared = unet._prepare_inputs(
            (first_latent_input_nhwc, first_time_input, alt_embeddings)
        )
        prompt_a_text_emb_uint16 = prompt_a_prepared["text_emb"]
        prompt_b_text_emb_uint16 = prompt_b_prepared["text_emb"]
        prompt_a_text_emb_dequant = append_quantization_stats(
            stats,
            "unet.text_emb.prompt_a",
            cond_embeddings,
            prompt_a_text_emb_uint16,
            text_emb_scale,
            text_emb_zero_point,
        )
        prompt_b_text_emb_dequant = append_quantization_stats(
            stats,
            "unet.text_emb.prompt_b",
            alt_embeddings,
            prompt_b_text_emb_uint16,
            text_emb_scale,
            text_emb_zero_point,
        )
        append_stat(
            stats,
            "unet.text_emb.prompt_a_minus_b.float32_before_quant",
            as_numpy(cond_embeddings) - as_numpy(alt_embeddings),
        )
        append_stat(
            stats,
            "unet.text_emb.prompt_a_minus_b.uint16_after_quant_signed",
            prompt_a_text_emb_uint16.astype(np.int32) - prompt_b_text_emb_uint16.astype(np.int32),
        )
        append_stat(
            stats,
            "unet.text_emb.prompt_a_minus_b.dequantized_after_quant",
            prompt_a_text_emb_dequant - prompt_b_text_emb_dequant,
        )
        prompt_a_noise = unet(first_latent_input_nhwc, first_time_input, cond_embeddings)
        prompt_b_noise = unet(first_latent_input_nhwc, first_time_input, alt_embeddings)
        append_stat(stats, "unet.prompt_compare.prompt_a.noise_nhwc", prompt_a_noise)
        append_stat(stats, "unet.prompt_compare.prompt_b.noise_nhwc", prompt_b_noise)
        append_stat(
            stats,
            "unet.prompt_compare.prompt_a_minus_b.noise_nchw",
            channel_first(prompt_a_noise) - channel_first(prompt_b_noise),
        )

        for index, timestep in enumerate(scheduler.timesteps):
            step = index + 1
            print(f"Step {step}/{args.num_steps}")
            time_input = torch.as_tensor([[timestep]], dtype=torch.float32)
            append_stat(stats, f"step_{step}.timestep", time_input)

            latent_input = scheduler.scale_model_input(latents, timestep)
            append_stat(stats, f"step_{step}.unet.input.latent_nchw", latent_input)
            latent_input_nhwc = channel_last(latent_input)
            append_stat(stats, f"step_{step}.unet.input.latent_nhwc", latent_input_nhwc)
            append_stat(stats, f"step_{step}.unet.input.cond_embedding", cond_embeddings)
            append_stat(stats, f"step_{step}.unet.input.uncond_embedding", uncond_embeddings)
            if step == 1:
                prepared_cond = unet._prepare_inputs(
                    (latent_input_nhwc, time_input, cond_embeddings)
                )
                prepared_uncond = unet._prepare_inputs(
                    (latent_input_nhwc, time_input, uncond_embeddings)
                )
                cond_text_emb_uint16 = prepared_cond["text_emb"]
                uncond_text_emb_uint16 = prepared_uncond["text_emb"]
                cond_text_emb_dequant = append_quantization_stats(
                    stats,
                    "step_1.unet.text_emb.cond",
                    cond_embeddings,
                    cond_text_emb_uint16,
                    text_emb_scale,
                    text_emb_zero_point,
                )
                uncond_text_emb_dequant = append_quantization_stats(
                    stats,
                    "step_1.unet.text_emb.uncond",
                    uncond_embeddings,
                    uncond_text_emb_uint16,
                    text_emb_scale,
                    text_emb_zero_point,
                )
                append_stat(
                    stats,
                    "step_1.unet.text_emb.cond_minus_uncond.float32_before_quant",
                    as_numpy(cond_embeddings) - as_numpy(uncond_embeddings),
                )
                append_stat(
                    stats,
                    "step_1.unet.text_emb.cond_minus_uncond.uint16_after_quant_signed",
                    cond_text_emb_uint16.astype(np.int32)
                    - uncond_text_emb_uint16.astype(np.int32),
                )
                append_stat(
                    stats,
                    "step_1.unet.text_emb.cond_minus_uncond.dequantized_after_quant",
                    cond_text_emb_dequant - uncond_text_emb_dequant,
                )

            noise_cond = unet(latent_input_nhwc, time_input, cond_embeddings)
            noise_uncond = unet(latent_input_nhwc, time_input, uncond_embeddings)
            append_stat(stats, f"step_{step}.unet.output.noise_cond_nhwc", noise_cond)
            append_stat(stats, f"step_{step}.unet.output.noise_uncond_nhwc", noise_uncond)

            noise_cond_nchw = channel_first(noise_cond)
            noise_uncond_nchw = channel_first(noise_uncond)
            noise_pred = noise_uncond_nchw + args.guidance_scale * (
                noise_cond_nchw - noise_uncond_nchw
            )
            append_stat(stats, f"step_{step}.noise_pred.nchw", noise_pred)
            append_stat(stats, f"step_{step}.noise_cond_minus_uncond.nchw", noise_cond_nchw - noise_uncond_nchw)

            latents = scheduler.step(noise_pred, timestep, latents).prev_sample
            append_stat(stats, f"step_{step}.latent.after_scheduler", latents)

    append_stat(stats, "vae.input.latent_nchw", latents)
    vae_latent_nhwc = channel_last(latents)
    append_stat(stats, "vae.input.latent_nhwc", vae_latent_nhwc)
    image = vae_decoder(vae_latent_nhwc)
    append_stat(stats, "vae.output.image_float", image)
    image_uint8 = to_uint8(np.asarray(image))[0]
    append_stat(stats, "image.before_save.uint8", image_uint8)
    Image.fromarray(image_uint8).save(output_dir / "diagnostic_image.png")

    print("Writing reports...")
    onnx_rows = collect_onnx_io(
        [MODEL_DIR / "text_encoder.onnx", MODEL_DIR / "unet.onnx", MODEL_DIR / "vae.onnx"]
    )
    write_onnx_io(output_dir / "onnx_io.md", onnx_rows)
    model_rows = collect_model_files(MODEL_DIR)
    write_model_files(output_dir / "model_files.md", MODEL_DIR, model_rows)

    cond_float_delta = find_stat(
        stats, "step_1.unet.text_emb.cond_minus_uncond.float32_before_quant"
    )
    cond_quant_delta = find_stat(
        stats, "step_1.unet.text_emb.cond_minus_uncond.uint16_after_quant_signed"
    )
    cond_dequant_delta = find_stat(
        stats, "step_1.unet.text_emb.cond_minus_uncond.dequantized_after_quant"
    )
    prompt_noise_delta = find_stat(stats, "unet.prompt_compare.prompt_a_minus_b.noise_nchw")
    prompt_text_delta = find_stat(
        stats, "unet.text_emb.prompt_a_minus_b.dequantized_after_quant"
    )
    wrapper_notes = [
        f"- UNet `text_emb` QDQ: dtype `{text_emb_dtype}`, scale `{text_emb_scale}`, zero_point `{text_emb_zero_point}`.",
        f"- cond/uncond text delta before quant std: `{fmt(cond_float_delta['std'] if cond_float_delta else None)}`.",
        f"- cond/uncond text delta after uint16 quant signed std: `{fmt(cond_quant_delta['std'] if cond_quant_delta else None)}`; zero ratio: `{fmt(cond_quant_delta['zero_ratio'] if cond_quant_delta else None)}`.",
        f"- cond/uncond text delta after dequant std: `{fmt(cond_dequant_delta['std'] if cond_dequant_delta else None)}`.",
        f"- very different prompt text delta after dequant std: `{fmt(prompt_text_delta['std'] if prompt_text_delta else None)}`.",
        f"- very different prompt UNet noise delta std: `{fmt(prompt_noise_delta['std'] if prompt_noise_delta else None)}`.",
    ]
    write_unet_text_emb_report(output_dir / "unet_text_emb.md", qdq_rows, wrapper_notes)

    if args.cpu_compare:
        notes.extend(
            [
                "# CPU Comparison",
                "",
                "CPU comparison was requested but skipped.",
                "",
                "Reason: these v0.48.0 assets are precompiled QNN context-wrapper ONNX files; they are intended for QNNExecutionProvider and are not a portable CPU ONNX baseline. Use non-context ONNX exports for a meaningful CPUExecutionProvider comparison.",
            ]
        )
        (output_dir / "cpu_compare.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    env = {
        "python": sys.version,
        "platform_machine": platform.machine(),
        "onnxruntime_version": onnxruntime.__version__,
        "onnxruntime_providers": onnxruntime.get_available_providers(),
        "prompt": args.prompt,
        "alt_prompt": args.alt_prompt,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
    }

    (output_dir / "tensor_stats.json").write_text(
        json.dumps({"environment": env, "stats": stats}, indent=2),
        encoding="utf-8",
    )
    write_tensor_stats_md(output_dir / "tensor_stats.md", env, stats)
    write_summary(output_dir / "summary.md", env, stats, notes)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.8g}"
    return str(value)


def write_tensor_stats_md(path: Path, env: dict[str, Any], stats: list[dict[str, Any]]) -> None:
    lines = [
        "# Tensor Statistics",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(env, indent=2),
        "```",
        "",
        "| name | shape | dtype | min | max | mean | std | nan | inf | zero_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stat in stats:
        lines.append(
            "| {name} | {shape} | {dtype} | {min} | {max} | {mean} | {std} | {nan_count} | {inf_count} | {zero_ratio} |".format(
                name=stat["name"],
                shape=stat["shape"],
                dtype=stat["dtype"],
                min=fmt(stat["min"]),
                max=fmt(stat["max"]),
                mean=fmt(stat["mean"]),
                std=fmt(stat["std"]),
                nan_count=fmt(stat["nan_count"]),
                inf_count=fmt(stat["inf_count"]),
                zero_ratio=fmt(stat["zero_ratio"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_stat(stats: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((item for item in stats if item["name"] == name), None)


def likely_issue(stats: list[dict[str, Any]]) -> str:
    text_delta = find_stat(stats, "text_encoder.output.cond_minus_uncond")
    image = find_stat(stats, "image.before_save.uint8")
    vae = find_stat(stats, "vae.output.image_float")
    final_latent = find_stat(stats, "vae.input.latent_nchw")
    first_noise = find_stat(stats, "step_1.noise_pred.nchw")
    first_guidance_delta = find_stat(stats, "step_1.noise_cond_minus_uncond.nchw")
    quant_text_delta = find_stat(
        stats, "step_1.unet.text_emb.cond_minus_uncond.uint16_after_quant_signed"
    )
    prompt_noise_delta = find_stat(stats, "unet.prompt_compare.prompt_a_minus_b.noise_nchw")

    observations = []
    if text_delta and (text_delta["std"] is not None) and text_delta["std"] < 1e-6:
        observations.append("conditional and unconditional text embeddings are nearly identical")
    if (
        first_noise
        and first_guidance_delta
        and first_noise["std"]
        and first_guidance_delta["std"] is not None
        and (first_guidance_delta["std"] / first_noise["std"]) < 0.02
    ):
        ratio = first_guidance_delta["std"] / first_noise["std"]
        observations.append(
            f"UNet conditional-unconditional noise delta is very small versus noise prediction (step 1 std ratio {ratio:.4f})"
        )
    if quant_text_delta and quant_text_delta["zero_ratio"] is not None:
        if quant_text_delta["zero_ratio"] > 0.95:
            observations.append("UNet text_emb cond/uncond delta is mostly zero after uint16 quantization")
        else:
            observations.append(
                f"UNet text_emb cond/uncond delta survives uint16 quantization (zero ratio {quant_text_delta['zero_ratio']:.4f})"
            )
    if prompt_noise_delta and prompt_noise_delta["std"] is not None:
        observations.append(
            f"very different prompts change UNet noise with std {prompt_noise_delta['std']:.6g}"
        )
    if final_latent and (final_latent["std"] is not None) and final_latent["std"] < 1e-6:
        observations.append("final latent is nearly constant before VAE")
    if vae and (vae["std"] is not None) and vae["std"] < 1e-6:
        observations.append("VAE output is nearly constant")
    if image and (image["std"] is not None) and image["std"] < 2:
        observations.append("saved uint8 image has very low variance")

    if observations:
        return "; ".join(observations)
    return "No single collapse point was obvious from simple variance checks; inspect tensor_stats.md for step-by-step drift."


def write_summary(
    path: Path,
    env: dict[str, Any],
    stats: list[dict[str, Any]],
    notes: list[str],
) -> None:
    key_names = [
        "text_encoder.output.cond_embedding",
        "text_encoder.output.uncond_embedding",
        "text_encoder.output.cond_minus_uncond",
        "step_1.unet.text_emb.cond_minus_uncond.float32_before_quant",
        "step_1.unet.text_emb.cond_minus_uncond.uint16_after_quant_signed",
        "step_1.unet.text_emb.cond_minus_uncond.dequantized_after_quant",
        "unet.prompt_compare.prompt_a_minus_b.noise_nchw",
        "latent.initial.scaled",
        "step_1.unet.output.noise_cond_nhwc",
        "step_1.noise_pred.nchw",
        "step_5.latent.after_scheduler",
        "vae.input.latent_nchw",
        "vae.output.image_float",
        "image.before_save.uint8",
    ]
    lines = [
        "# Intermediate Debug Summary",
        "",
        "## Environment",
        "",
        f"- Python architecture: `{env['platform_machine']}`",
        f"- ONNX Runtime: `{env['onnxruntime_version']}`",
        f"- Providers: `{env['onnxruntime_providers']}`",
        f"- Prompt: `{env['prompt']}`",
        f"- Alt prompt: `{env.get('alt_prompt', '')}`",
        f"- Steps: `{env['num_steps']}`",
        f"- Seed: `{env['seed']}`",
        "",
        "## Key Statistics",
        "",
        "| tensor | shape | dtype | min | max | mean | std | zero_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in key_names:
        stat = find_stat(stats, name)
        if not stat:
            continue
        lines.append(
            f"| {name} | {stat['shape']} | {stat['dtype']} | {fmt(stat['min'])} | {fmt(stat['max'])} | {fmt(stat['mean'])} | {fmt(stat['std'])} | {fmt(stat['zero_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## Likely Collapse Point",
            "",
            likely_issue(stats),
            "",
            "## Notes",
            "",
            "- v0.56.0 was not used.",
            "- The script uses the existing v0.48.0 `model/` directory and `onnxruntime-qnn==1.24.1` environment.",
            "- `diagnostic_image.png` is produced from the same 5-step diagnostic path.",
        ]
    )
    if notes:
        lines.extend(["", "## CPU Comparison", ""])
        lines.extend(notes[2:80] if notes[:2] == ["# CPU Comparison", ""] else notes[:80])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--alt-prompt", default=DEFAULT_ALT_PROMPT)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--output-dir", default="outputs/intermediate_debug")
    parser.add_argument("--cpu-compare", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_diagnostics(parse_args())
