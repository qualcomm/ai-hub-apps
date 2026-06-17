# UNet Text Embedding Input Debug

## OnnxModelTorchWrapper._prepare_inputs()

The wrapper prepares each input by converting tensors to numpy arrays, then quantizes float inputs when the ONNX input is an integer tensor with QDQ parameters and the input name is enabled in `quantize_user_input`.

Quantization formula observed from the installed `qai_hub_models==0.48.0` wrapper:

```text
uint_value = round(float_value / scale) + zero_point
uint_value = clip(uint_value, dtype_min, dtype_max).astype(dtype)
```

## Metadata / Wrapper QDQ

| source | input | dtype | scale | zero_point |
| --- | --- | --- | --- | --- |
| metadata.yaml / OnnxModelTorchWrapper | unet.text_emb | uint16 | 0.00034632044844329357 | 23638 |

## Findings

- UNet `text_emb` QDQ: dtype `uint16`, scale `0.00034632044844329357`, zero_point `23638`.
- cond/uncond text delta before quant std: `0.69729782`.
- cond/uncond text delta after uint16 quant signed std: `2013.4483`; zero ratio: `0.013278713`.
- cond/uncond text delta after dequant std: `0.69729833`.
- very different prompt text delta after dequant std: `0.66667875`.
- very different prompt UNet noise delta std: `0.017968048`.
