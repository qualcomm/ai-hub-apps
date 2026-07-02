// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.superresolution;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.util.Pair;

import com.quicinc.ImageProcessing;
import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.image.ColorSpaceType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map;

public class SuperResolution implements AutoCloseable {
    private static final String TAG = "ImageClassification";
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final int[] inputShape;
    private final DataType inputType;
    private final DataType outputType;
    private long preprocessingTime;
    private long postprocessingTime;
    private final ImageProcessor inputImageProcessor;
    private final ImageProcessor outputImageProcessor;
    private final TensorBuffer outputBuffer;
    private final Map<Integer, Object> outputBindings;
    private final TensorImage outputImage;

    /**
     * Create an Image Classifier from the given model.
     * Uses default compute units: NPU, GPU, CPU.
     * Ignores compute units that fail to load.
     *
     * @param context    App context.
     * @param modelPath  Model path to load.
     * @throws IOException If the model can't be read from disk.
     */
    public SuperResolution(Context context,
                           String modelPath) throws IOException, NoSuchAlgorithmException {
        this(context, modelPath, AIHubDefaults.delegatePriorityOrder);
    }

    /**
     * Create an Image Classifier from the given model.
     * Ignores compute units that fail to load.
     *
     * @param context     App context.
     * @param modelPath   Model path to load.
     * @param delegatePriorityOrder Priority order of delegate sets to enable.
     * @throws IOException If the model can't be read from disk.
     */
    public SuperResolution(Context context,
                           String modelPath,
                           TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;

        // Validate TF Lite model fits requirements for this app
        assert tfLiteInterpreter.getInputTensorCount() == 1;
        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        inputShape = inputTensor.shape();
        inputType = inputTensor.dataType();
        assert inputShape.length == 4; // 4D Input Tensor: [Batch, Height, Width, Channels]
        assert inputShape[0] == 1; // Batch size is 1
        assert inputShape[3] == 3; // Input tensor should have 3 channels
        assert inputType == DataType.UINT8 || inputType == DataType.FLOAT32; // UINT8 (Quantized) and FP32 Input Supported

        assert tfLiteInterpreter.getOutputTensorCount() == 1;
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        outputType = outputTensor.dataType();
        assert outputShape.length == 4; // 4D Output Tensor: [Batch, Height, Width, Channels]
        assert outputShape[0] == 1; // Batch size is 1
        assert outputShape[3] == 3; // Output tensor should have 3 channels
        assert outputType == DataType.UINT8 || inputType == DataType.FLOAT32; // UINT8 (Quantized) and FP32 Input Supported

        // Set-up preprocessor
        inputImageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
        outputImageProcessor = new ImageProcessor.Builder()
                .add(new NormalizeOp(0.0f, 1 / 255.0f))
                .add(new CastOp(DataType.UINT8))
                .build();

        // Set-up output image
        outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType);
        outputBindings = new HashMap<>();
        outputBindings.put(0, outputBuffer.getBuffer());
        outputImage = new TensorImage(outputType);
        outputImage.load(outputBuffer, ColorSpaceType.RGB);
    }

    /**
     * Free resources used by the classifier.
     */
    @Override
    public void close() {
        tfLiteInterpreter.close();
        for (Delegate delegate: tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }

    /**
     * @return last preprocessing time in microseconds.
     */
    public long getLastPreprocessingTime() {
        if (preprocessingTime == 0) {
            throw new RuntimeException("Cannot get preprocessing time as model has not yet been executed.");
        }
        return preprocessingTime;
    }

    /**
     * @return last inference time in microseconds.
     */
    public long getLastInferenceTime() {
        return tfLiteInterpreter.getLastNativeInferenceDurationNanoseconds();
    }

    /**
     * @return last postprocessing time in microseconds.
     */
    public long getLastPostprocessingTime() {
        if (postprocessingTime == 0) {
            throw new RuntimeException("Cannot get postprocessing time as model has not yet been executed.");
        }
        return postprocessingTime;
    }

    /** Model input height and width. **/
    public int[] getInputWidthHeight() {
        return new int[] {inputShape[1], inputShape[2]};
    }

    /**
     * Preprocess using the provided image (resize, convert to model input data type).
     * Sets the input buffer held by this.tfLiteModel to the processed input.
     *
     * @param image RGBA-8888 Bitmap to preprocess.
     * @return Array of inputs to pass to the interpreter.
     */
    private ByteBuffer[] preprocess(Bitmap image) {
        long prepStartTime = System.nanoTime();
        Bitmap resizedImg;

        // Resize input image
        if (image.getHeight() > inputShape[1] || image.getWidth() > inputShape[2]) {
            // This image is larger than the model's desired input size.
            // While this app could easily resize the large image to fit, that defeats the purpose of super resolution.
            throw new RuntimeException("Input image is too big for this model. Expected Width of " + inputShape[1] + " and Height of " + inputShape[2]);
        } else if (image.getHeight() != inputShape[1] || image.getWidth() != inputShape[2]) {
            resizedImg = ImageProcessing.resizeAndPadMaintainAspectRatio(image, inputShape[1], inputShape[2], 0xFF);
        } else {
            resizedImg = image;
        }

        // Convert type and fill input buffer
        ByteBuffer inputBuffer;
        TensorImage tImg = TensorImage.fromBitmap(resizedImg);
        if (inputType == DataType.FLOAT32) {
            // Divide float values by 255
            inputBuffer = inputImageProcessor.process(tImg).getBuffer();
        } else {
            inputBuffer = tImg.getTensorBuffer().getBuffer();
        }

        preprocessingTime = System.nanoTime() - prepStartTime;
        Log.d(TAG, "Preprocessing Time: " + preprocessingTime / 1000000 + " ms");

        return new ByteBuffer[] {inputBuffer};
    }


    /**
     * Reads the output buffer on tfLiteModel and processes it into a readable output image.
     *
     * @return Upscaled image, in RGBA-8888 format.
     */
    private Bitmap postprocess() {
        long postStartTime = System.nanoTime();

        TensorImage img = outputImage;
        if (outputType == DataType.FLOAT32) {
            // Multiply float values by 255
            img = outputImageProcessor.process(outputImage);
        }
        Bitmap bitmap = img.getBitmap();

        postprocessingTime = System.nanoTime() - postStartTime;
        Log.d(TAG, "Postprocessing Time: " + postprocessingTime / 1000000 + " ms");

        return bitmap;
    }

    /**
     * Upscale the provided input image.
     *
     * @param image RGBA-8888 bitmap image to upscale.
     * @return Predicted, upscaled image, in RGBA-8888 format.
     */
    public Bitmap generateUpscaledImage(Bitmap image) {
        // Preprocessing: Resize, convert type
        ByteBuffer[] inputs = preprocess(image);

        // Inference
        outputBuffer.getBuffer().clear();
        tfLiteInterpreter.runForMultipleInputsOutputs(inputs, outputBindings);

        // Postprocessing: Compute top K indices and convert to labels
        return postprocess();
    }
}
