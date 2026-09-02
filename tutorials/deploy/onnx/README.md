## Deployable ONNX Asset

Users can use the script `build_deployable_asset.py` to convert their quantized ONNX models to the deployable asset for their applications.

## Future Work

AI Hub's profile and inference jobs, the ONNX graph (if it is quantized) would be transformed to op-centric quantized representation that has a one-to-one mapping with a QOp representation. The weights would be stored quantized which helps map cleanly to QOp. This leads to better performance on device and reduces the memory footprint of the model.

### Example to run the script

1. Install the dependencies to run the script.

```bash
cd deploy/onnx && pip install -r requirements.txt
```

1. Export AI Hub Model which uses Quantize Jobs and target ONNX as the runtime.

```bash
python qai_hub_models/models/googlenet/export.py --target-runtime onnx --quantize
```

The export script downloads the compiled asset to your local machine.

1. Use the compiled asset and convert to a deployable ONNX asset.

```bash
python build_deployable_asset.py -f /path/to/compiled/onnx/asset
```

The modified model will be saved in the same path provided. This ONNX model can be moved to your application and deployed.
