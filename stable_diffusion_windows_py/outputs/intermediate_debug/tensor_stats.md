# Tensor Statistics

## Environment

```json
{
  "python": "3.11.15 | packaged by Anaconda, Inc. | (main, Jun 11 2026, 15:12:53) [MSC v.1942 64 bit (AMD64)]",
  "platform_machine": "AMD64",
  "onnxruntime_version": "1.24.1",
  "onnxruntime_providers": [
    "QNNExecutionProvider",
    "AzureExecutionProvider",
    "CPUExecutionProvider"
  ],
  "prompt": "A girl taking a walk at sunset",
  "alt_prompt": "A red sports car parked in a snowy mountain at night",
  "seed": 47,
  "num_steps": 5,
  "guidance_scale": 7.5
}
```

| name | shape | dtype | min | max | mean | std | nan | inf | zero_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text_encoder.input.tokens | [1, 77] | int32 | 0 | 49407 | 1421 | 7855.0557 | 0 | 0 | 0.88311688 |
| text_encoder.input.uncond_tokens | [1, 77] | int32 | 0 | 49407 | 1283.2857 | 7858.488 | 0 | 0 | 0.97402597 |
| text_encoder.input.alt_tokens | [1, 77] | int32 | 0 | 49407 | 1824.6364 | 8157.2071 | 0 | 0 | 0.83116883 |
| text_encoder.output.cond_embedding | [1, 77, 1024] | float32 | -6.6315475 | 12.992723 | -0.16970948 | 1.0241217 | 0 | 0 | 0.00011414367 |
| text_encoder.output.uncond_embedding | [1, 77, 1024] | float32 | -5.7696242 | 12.992723 | -0.16814759 | 1.0403697 | 0 | 0 | 0.00020292208 |
| text_encoder.output.alt_embedding | [1, 77, 1024] | float32 | -6.981761 | 12.992723 | -0.16791664 | 1.0214604 | 0 | 0 | 8.8778409e-05 |
| text_encoder.output.cond_minus_uncond | [1, 77, 1024] | float32 | -9.3341761 | 8.0494576 | -0.0015618909 | 0.69729782 | 0 | 0 | 0.013278713 |
| text_encoder.output.cond_minus_alt | [1, 77, 1024] | float32 | -8.0407467 | 12.914334 | -0.0017928369 | 0.66667773 | 0 | 0 | 0.026202313 |
| latent.initial.randn | [1, 4, 64, 64] | float32 | -4.1334796 | 3.86008 | 0.0074700304 | 0.993912 | 0 | 0 | 0 |
| latent.initial.scaled | [1, 4, 64, 64] | float32 | -4.1334796 | 3.86008 | 0.0074700304 | 0.993912 | 0 | 0 | 0 |
| unet.text_emb.prompt_a.float32_before_quant | [1, 77, 1024] | float32 | -6.6315475 | 12.992723 | -0.16970948 | 1.0241217 | 0 | 0 | 0.00011414367 |
| unet.text_emb.prompt_a.uint16_after_quant | [1, 77, 1024] | uint16 | 4489 | 61154 | 23147.963 | 2957.1508 | 0 | 0 | 0 |
| unet.text_emb.prompt_a.dequantized_after_quant | [1, 77, 1024] | float32 | -6.6316905 | 12.992558 | -0.16970992 | 1.0241218 | 0 | 0 | 0.00011414367 |
| unet.text_emb.prompt_a.quantization_abs_error | [1, 77, 1024] | float32 | 0 | 0.00017312914 | 8.6599969e-05 | 4.9936748e-05 | 0 | 0 | 0.00024096997 |
| unet.text_emb.prompt_b.float32_before_quant | [1, 77, 1024] | float32 | -6.981761 | 12.992723 | -0.16791664 | 1.0214604 | 0 | 0 | 8.8778409e-05 |
| unet.text_emb.prompt_b.uint16_after_quant | [1, 77, 1024] | uint16 | 3478 | 61154 | 23153.141 | 2949.4665 | 0 | 0 | 0 |
| unet.text_emb.prompt_b.dequantized_after_quant | [1, 77, 1024] | float32 | -6.9818201 | 12.992558 | -0.16791651 | 1.0214606 | 0 | 0 | 8.8778409e-05 |
| unet.text_emb.prompt_b.quantization_abs_error | [1, 77, 1024] | float32 | 0 | 0.00017312914 | 8.6806189e-05 | 5.0074136e-05 | 0 | 0 | 0.00022828734 |
| unet.text_emb.prompt_a_minus_b.float32_before_quant | [1, 77, 1024] | float32 | -8.0407467 | 12.914334 | -0.0017928369 | 0.66667773 | 0 | 0 | 0.026202313 |
| unet.text_emb.prompt_a_minus_b.uint16_after_quant_signed | [1, 77, 1024] | int32 | -23218 | 37291 | -5.17847 | 1925.0343 | 0 | 0 | 0.026202313 |
| unet.text_emb.prompt_a_minus_b.dequantized_after_quant | [1, 77, 1024] | float32 | -8.0408678 | 12.914636 | -0.0017934101 | 0.66667875 | 0 | 0 | 0.026202313 |
| unet.prompt_compare.prompt_a.noise_nhwc | [1, 64, 64, 4] | float32 | -4.1717653 | 3.8935988 | 0.0023330584 | 0.99883529 | 0 | 0 | 0.00012207031 |
| unet.prompt_compare.prompt_b.noise_nhwc | [1, 64, 64, 4] | float32 | -4.1782088 | 3.8878918 | 0.0065542202 | 0.99853757 | 0 | 0 | 0 |
| unet.prompt_compare.prompt_a_minus_b.noise_nchw | [1, 4, 64, 64] | float32 | -0.16292369 | 0.046575904 | -0.0042211618 | 0.017968048 | 0 | 0 | 0.0054931641 |
| step_1.timestep | [1, 1] | float32 | 801 | 801 | 801 | 0 | 0 | 0 | 0 |
| step_1.unet.input.latent_nchw | [1, 4, 64, 64] | float32 | -4.1334796 | 3.86008 | 0.0074700304 | 0.993912 | 0 | 0 | 0 |
| step_1.unet.input.latent_nhwc | [1, 64, 64, 4] | float32 | -4.1334796 | 3.86008 | 0.0074700304 | 0.993912 | 0 | 0 | 0 |
| step_1.unet.input.cond_embedding | [1, 77, 1024] | float32 | -6.6315475 | 12.992723 | -0.16970948 | 1.0241217 | 0 | 0 | 0.00011414367 |
| step_1.unet.input.uncond_embedding | [1, 77, 1024] | float32 | -5.7696242 | 12.992723 | -0.16814759 | 1.0403697 | 0 | 0 | 0.00020292208 |
| step_1.unet.text_emb.cond.float32_before_quant | [1, 77, 1024] | float32 | -6.6315475 | 12.992723 | -0.16970948 | 1.0241217 | 0 | 0 | 0.00011414367 |
| step_1.unet.text_emb.cond.uint16_after_quant | [1, 77, 1024] | uint16 | 4489 | 61154 | 23147.963 | 2957.1508 | 0 | 0 | 0 |
| step_1.unet.text_emb.cond.dequantized_after_quant | [1, 77, 1024] | float32 | -6.6316905 | 12.992558 | -0.16970992 | 1.0241218 | 0 | 0 | 0.00011414367 |
| step_1.unet.text_emb.cond.quantization_abs_error | [1, 77, 1024] | float32 | 0 | 0.00017312914 | 8.6599969e-05 | 4.9936748e-05 | 0 | 0 | 0.00024096997 |
| step_1.unet.text_emb.uncond.float32_before_quant | [1, 77, 1024] | float32 | -5.7696242 | 12.992723 | -0.16814759 | 1.0403697 | 0 | 0 | 0.00020292208 |
| step_1.unet.text_emb.uncond.uint16_after_quant | [1, 77, 1024] | uint16 | 6978 | 61154 | 23152.473 | 3004.0659 | 0 | 0 | 0 |
| step_1.unet.text_emb.uncond.dequantized_after_quant | [1, 77, 1024] | float32 | -5.7696986 | 12.992558 | -0.16814798 | 1.0403694 | 0 | 0 | 0.00020292208 |
| step_1.unet.text_emb.uncond.quantization_abs_error | [1, 77, 1024] | float32 | 0 | 0.00017312914 | 8.6778316e-05 | 5.0016647e-05 | 0 | 0 | 0.00030438312 |
| step_1.unet.text_emb.cond_minus_uncond.float32_before_quant | [1, 77, 1024] | float32 | -9.3341761 | 8.0494576 | -0.0015618909 | 0.69729782 | 0 | 0 | 0.013278713 |
| step_1.unet.text_emb.cond_minus_uncond.uint16_after_quant_signed | [1, 77, 1024] | int32 | -26952 | 23243 | -4.5100954 | 2013.4483 | 0 | 0 | 0.013278713 |
| step_1.unet.text_emb.cond_minus_uncond.dequantized_after_quant | [1, 77, 1024] | float32 | -9.3340292 | 8.0495262 | -0.0015619383 | 0.69729833 | 0 | 0 | 0.013278713 |
| step_1.unet.output.noise_cond_nhwc | [1, 64, 64, 4] | float32 | -4.1717653 | 3.8935988 | 0.0023330584 | 0.99883529 | 0 | 0 | 0.00012207031 |
| step_1.unet.output.noise_uncond_nhwc | [1, 64, 64, 4] | float32 | -4.1704769 | 3.8877077 | 0.0031742033 | 0.99741883 | 0 | 0 | 0.00012207031 |
| step_1.noise_pred.nchw | [1, 4, 64, 64] | float32 | -4.18014 | 3.931891 | -0.0031343835 | 1.0100011 | 0 | 0 | 0 |
| step_1.noise_cond_minus_uncond.nchw | [1, 4, 64, 64] | float32 | -0.052651167 | 0.049153253 | -0.00084114489 | 0.0091155551 | 0 | 0 | 0.0087890625 |
| step_1.latent.after_scheduler | [1, 4, 64, 64] | float32 | -3.1270456 | 2.9140821 | 0.0079724382 | 0.75177003 | 0 | 0 | 0 |
| step_2.timestep | [1, 1] | float32 | 601 | 601 | 601 | 0 | 0 | 0 | 0 |
| step_2.unet.input.latent_nchw | [1, 4, 64, 64] | float32 | -3.1270456 | 2.9140821 | 0.0079724382 | 0.75177003 | 0 | 0 | 0 |
| step_2.unet.input.latent_nhwc | [1, 64, 64, 4] | float32 | -3.1270456 | 2.9140821 | 0.0079724382 | 0.75177003 | 0 | 0 | 0 |
| step_2.unet.input.cond_embedding | [1, 77, 1024] | float32 | -6.6315475 | 12.992723 | -0.16970948 | 1.0241217 | 0 | 0 | 0.00011414367 |
| step_2.unet.input.uncond_embedding | [1, 77, 1024] | float32 | -5.7696242 | 12.992723 | -0.16814759 | 1.0403697 | 0 | 0 | 0.00020292208 |
| step_2.unet.output.noise_cond_nhwc | [1, 64, 64, 4] | float32 | -3.6693716 | 3.3383698 | -0.00061920642 | 0.86980318 | 0 | 0 | 0.00018310547 |
| step_2.unet.output.noise_uncond_nhwc | [1, 64, 64, 4] | float32 | -3.6500418 | 3.3426039 | -0.0010572513 | 0.86946045 | 0 | 0 | 6.1035156e-05 |
| step_2.noise_pred.nchw | [1, 4, 64, 64] | float32 | -3.7950153 | 3.3108482 | 0.0022280852 | 0.8734602 | 0 | 0 | 0 |
| step_2.noise_cond_minus_uncond.nchw | [1, 4, 64, 64] | float32 | -0.05780568 | 0.041237175 | 0.00043804485 | 0.007161856 | 0 | 0 | 0.012268066 |
| step_2.latent.after_scheduler | [1, 4, 64, 64] | float32 | -1.8808618 | 1.8187597 | 0.00697266 | 0.46447138 | 0 | 0 | 0 |
| step_3.timestep | [1, 1] | float32 | 401 | 401 | 401 | 0 | 0 | 0 | 0 |
| step_3.unet.input.latent_nchw | [1, 4, 64, 64] | float32 | -1.8808618 | 1.8187597 | 0.00697266 | 0.46447138 | 0 | 0 | 0 |
| step_3.unet.input.latent_nhwc | [1, 64, 64, 4] | float32 | -1.8808618 | 1.8187597 | 0.00697266 | 0.46447138 | 0 | 0 | 0 |
| step_3.unet.input.cond_embedding | [1, 77, 1024] | float32 | -6.6315475 | 12.992723 | -0.16970948 | 1.0241217 | 0 | 0 | 0.00011414367 |
| step_3.unet.input.uncond_embedding | [1, 77, 1024] | float32 | -5.7696242 | 12.992723 | -0.16814759 | 1.0403697 | 0 | 0 | 0.00020292208 |
| step_3.unet.output.noise_cond_nhwc | [1, 64, 64, 4] | float32 | -3.2363813 | 2.8422351 | -0.0011052973 | 0.74890468 | 0 | 0 | 6.1035156e-05 |
| step_3.unet.output.noise_uncond_nhwc | [1, 64, 64, 4] | float32 | -3.2308586 | 2.8542011 | -0.0013359098 | 0.7485473 | 0 | 0 | 0.00012207031 |
| step_3.noise_pred.nchw | [1, 4, 64, 64] | float32 | -3.272279 | 2.8522682 | 0.00039368462 | 0.75190856 | 0 | 0 | 0 |
| step_3.noise_cond_minus_uncond.nchw | [1, 4, 64, 64] | float32 | -0.033873387 | 0.030007407 | 0.00023061256 | 0.0045959092 | 0 | 0 | 0.02166748 |
| step_3.latent.after_scheduler | [1, 4, 64, 64] | float32 | -0.67559671 | 0.78725356 | 0.0064376108 | 0.18741077 | 0 | 0 | 0 |
| step_4.timestep | [1, 1] | float32 | 201 | 201 | 201 | 0 | 0 | 0 | 0 |
| step_4.unet.input.latent_nchw | [1, 4, 64, 64] | float32 | -0.67559671 | 0.78725356 | 0.0064376108 | 0.18741077 | 0 | 0 | 0 |
| step_4.unet.input.latent_nhwc | [1, 64, 64, 4] | float32 | -0.67559671 | 0.78725356 | 0.0064376108 | 0.18741077 | 0 | 0 | 0 |
| step_4.unet.input.cond_embedding | [1, 77, 1024] | float32 | -6.6315475 | 12.992723 | -0.16970948 | 1.0241217 | 0 | 0 | 0.00011414367 |
| step_4.unet.input.uncond_embedding | [1, 77, 1024] | float32 | -5.7696242 | 12.992723 | -0.16814759 | 1.0403697 | 0 | 0 | 0.00020292208 |
| step_4.unet.output.noise_cond_nhwc | [1, 64, 64, 4] | float32 | -2.7905045 | 2.4326248 | 0.0008677523 | 0.61938819 | 0 | 0 | 0.00012207031 |
| step_4.unet.output.noise_uncond_nhwc | [1, 64, 64, 4] | float32 | -2.7901363 | 2.4374113 | 0.00072284953 | 0.62070421 | 0 | 0 | 0.00012207031 |
| step_4.noise_pred.nchw | [1, 4, 64, 64] | float32 | -2.7928972 | 2.4015126 | 0.0018096202 | 0.61163766 | 0 | 0 | 0 |
| step_4.noise_cond_minus_uncond.nchw | [1, 4, 64, 64] | float32 | -0.04215765 | 0.041421235 | 0.00014490276 | 0.0046779277 | 0 | 0 | 0.02130127 |
| step_4.latent.after_scheduler | [1, 4, 64, 64] | float32 | -0.52162111 | 0.6899755 | 0.0048752632 | 0.12272436 | 0 | 0 | 0 |
| step_5.timestep | [1, 1] | float32 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| step_5.unet.input.latent_nchw | [1, 4, 64, 64] | float32 | -0.52162111 | 0.6899755 | 0.0048752632 | 0.12272436 | 0 | 0 | 0 |
| step_5.unet.input.latent_nhwc | [1, 64, 64, 4] | float32 | -0.52162111 | 0.6899755 | 0.0048752632 | 0.12272436 | 0 | 0 | 0 |
| step_5.unet.input.cond_embedding | [1, 77, 1024] | float32 | -6.6315475 | 12.992723 | -0.16970948 | 1.0241217 | 0 | 0 | 0.00011414367 |
| step_5.unet.input.uncond_embedding | [1, 77, 1024] | float32 | -5.7696242 | 12.992723 | -0.16814759 | 1.0403697 | 0 | 0 | 0.00020292208 |
| step_5.unet.output.noise_cond_nhwc | [1, 64, 64, 4] | float32 | -2.6286855 | 2.6412039 | 0.001855125 | 0.55710908 | 0 | 0 | 0.00012207031 |
| step_5.unet.output.noise_uncond_nhwc | [1, 64, 64, 4] | float32 | -2.6507769 | 2.7072937 | 0.0017829773 | 0.56114531 | 0 | 0 | 0.00030517578 |
| step_5.noise_pred.nchw | [1, 4, 64, 64] | float32 | -2.4850914 | 2.2116199 | 0.002324085 | 0.53557107 | 0 | 0 | 0 |
| step_5.noise_cond_minus_uncond.nchw | [1, 4, 64, 64] | float32 | -0.073453709 | 0.081922054 | 7.2147667e-05 | 0.010911129 | 0 | 0 | 0.0090332031 |
| step_5.latent.after_scheduler | [1, 4, 64, 64] | float32 | -0.50596952 | 0.66309458 | 0.0048467099 | 0.1170036 | 0 | 0 | 0 |
| vae.input.latent_nchw | [1, 4, 64, 64] | float32 | -0.50596952 | 0.66309458 | 0.0048467099 | 0.1170036 | 0 | 0 | 0 |
| vae.input.latent_nhwc | [1, 64, 64, 4] | float32 | -0.50596952 | 0.66309458 | 0.0048467099 | 0.1170036 | 0 | 0 | 0 |
| vae.output.image_float | [1, 512, 512, 3] | float32 | 0.26594949 | 0.5824216 | 0.44006879 | 0.054100643 | 0 | 0 | 0 |
| image.before_save.uint8 | [512, 512, 3] | uint8 | 68 | 149 | 112.21487 | 13.800269 | 0 | 0 | 0 |
