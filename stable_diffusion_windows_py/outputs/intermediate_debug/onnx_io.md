# ONNX I/O

| model | direction | name | shape | dtype |
| --- | --- | --- | --- | --- |
| text_encoder.onnx | input | tokens | [1, 77] | INT32 |
| text_encoder.onnx | output | text_embedding | [1, 77, 1024] | UINT16 |
| unet.onnx | input | latent | [1, 64, 64, 4] | UINT16 |
| unet.onnx | input | timestep | [1, 1] | UINT16 |
| unet.onnx | input | text_emb | [1, 77, 1024] | UINT16 |
| unet.onnx | output | output_latent | [1, 64, 64, 4] | UINT16 |
| vae.onnx | input | latent | [1, 64, 64, 4] | UINT16 |
| vae.onnx | output | image | [1, 512, 512, 3] | UINT16 |
