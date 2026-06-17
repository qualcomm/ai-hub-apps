# Model Files

Model directory: `model`

v0.48.0 metadata match: `True`

| file | exists | size_bytes |
| --- | --- | --- |
| metadata.yaml | True | 1480 |
| text_encoder.onnx | True | 733 |
| text_encoder_qairt_context.bin | True | 396058624 |
| unet.onnx | True | 1714 |
| unet_qairt_context.bin | True | 882139136 |
| vae.onnx | True | 873 |
| vae_qairt_context.bin | True | 64622592 |

## metadata.yaml

```yaml
runtime: precompiled_qnn_onnx
precision: w8a16
tool_versions:
  qairt: 2.42.0.251225135753_193295
  onnx_runtime: 1.24.1
model_files:
  text_encoder.onnx:
    inputs:
      tokens:
        shape: [1, 77]
        dtype: int32
    outputs:
      text_embedding:
        shape: [1, 77, 1024]
        dtype: uint16
        quantization_parameters:
          scale: 0.00036291510332375765
          zero_point: 25915
  unet.onnx:
    inputs:
      latent:
        shape: [1, 64, 64, 4]
        dtype: uint16
        quantization_parameters:
          scale: 0.0003279144875705242
          zero_point: 34760
      timestep:
        shape: [1, 1]
        dtype: uint16
        quantization_parameters:
          scale: 0.014770733192563057
          zero_point: 0
      text_emb:
        shape: [1, 77, 1024]
        dtype: uint16
        quantization_parameters:
          scale: 0.00034632044844329357
          zero_point: 23638
    outputs:
      output_latent:
        shape: [1, 64, 64, 4]
        dtype: uint16
        quantization_parameters:
          scale: 0.00018409450422041118
          zero_point: 30388
  vae.onnx:
    inputs:
      latent:
        shape: [1, 64, 64, 4]
        dtype: uint16
        quantization_parameters:
          scale: 0.0003278455988038331
          zero_point: 34752
    outputs:
      image:
        shape: [1, 512, 512, 3]
        dtype: uint16
        quantization_parameters:
          scale: 1.5259021893143654e-05
          zero_point: 0
```
