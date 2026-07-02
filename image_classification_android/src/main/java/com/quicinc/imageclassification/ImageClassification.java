// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.imageclassification;

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
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.stream.Collectors;

public class ImageClassification implements AutoCloseable {
    private static final String TAG = "ImageClassification";
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final List<String> labelList;
    private final int[] inputShape;
    private final DataType inputType;
    private final DataType outputType;
    private long preprocessingTime;
    private long postprocessingTime;
    private static final int TOPK = 3;
    private final ImageProcessor imageProcessor;

    /**
     * Create an Image Classifier from the given model.
     * Uses default compute units: NPU, GPU, CPU.
     * Ignores compute units that fail to load.
     *
     * @param context    App context.
     * @param modelPath  Model path to load.
     * @param labelsPath Labels path to load.
     * @throws IOException If the model can't be read from disk.
     */
    public ImageClassification(Context context,
                               String modelPath,
                               String labelsPath) throws IOException, NoSuchAlgorithmException {
        this(context, modelPath, labelsPath, AIHubDefaults.delegatePriorityOrder);
    }

    /**
     * Create an Image Classifier from the given model.
     * Ignores compute units that fail to load.
     *
     * @param context     App context.
     * @param modelPath   Model path to load.
     * @param labelsPath  Labels path to load.
     * @param delegatePriorityOrder Priority order of delegate sets to enable.
     * @throws IOException If the model can't be read from disk.
     */
    public ImageClassification(Context context,
                               String modelPath,
                               String labelsPath,
                               TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Load labels
        try (BufferedReader labelsFile = new BufferedReader(new InputStreamReader(context.getAssets().open(labelsPath)))) {
            labelList = labelsFile.lines().collect(Collectors.toCollection(ArrayList::new));
        }

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
        assert inputType == DataType.UINT8 || inputType == DataType.FLOAT32; // INT8 (Quantized) and FP32 Input Supported

        assert tfLiteInterpreter.getOutputTensorCount() == 1;
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        outputType = outputTensor.dataType();
        assert outputShape.length == 2; // 2D Output Tensor: [Batch, # of Labels]
        assert inputShape[0] == 1; // Batch size is 1
        assert outputShape[1] == labelList.size(); // # of labels == output dim
        assert outputType == DataType.UINT8 || outputType == DataType.INT8 | outputType == DataType.FLOAT32; // U/INT8 (Quantized) and FP32 Output Supported

        // Set-up preprocessor
        imageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
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
        if (image.getHeight() != inputShape[1] || image.getWidth() != inputShape[2]) {
            resizedImg = ImageProcessing.resizeAndPadMaintainAspectRatio(image, inputShape[1], inputShape[2], 0);
        } else {
            resizedImg = image;
        }

        // Convert type and fill input buffer
        ByteBuffer inputBuffer;
        TensorImage tImg = TensorImage.fromBitmap(resizedImg);
        if (inputType == DataType.FLOAT32) {
            inputBuffer = imageProcessor.process(tImg).getBuffer();
        } else {
            inputBuffer = tImg.getTensorBuffer().getBuffer();
        }

        preprocessingTime = System.nanoTime() - prepStartTime;
        Log.d(TAG, "Preprocessing Time: " + preprocessingTime / 1000000 + " ms");

        return new ByteBuffer[] {inputBuffer};
    }


    /**
     * Reads the output buffers on tfLiteModel and processes them into readable output classes.
     *
     * @return Predicted object class names, in order of confidence (highest confidence first).
     */
    private ArrayList<String> postprocess() {
        long postStartTime = System.nanoTime();

        List<Integer> indexList;
        ByteBuffer outputBuffer = tfLiteInterpreter.getOutputTensor(0).asReadOnlyBuffer();
        if (outputType == DataType.FLOAT32) {
            indexList = findTopKFloatIndices(outputBuffer.asFloatBuffer(), TOPK);
        } else {
            indexList = findTopKByteIndices(outputBuffer, TOPK);
        }
        ArrayList<String> labels = indexList.stream().map(labelList::get).collect(Collectors.toCollection(ArrayList<String>::new));

        postprocessingTime = System.nanoTime() - postStartTime;
        Log.d(TAG, "Postprocessing Time: " + postprocessingTime / 1000000 + " ms");

        return labels;
    }

    /**
     * Predict the most likely classes of the object in the image.
     *
     * @param image RGBA-8888 bitmap image to predict class of.
     * @return Predicted object class names, in order of confidence (highest confidence first).
     */
    public ArrayList<String> predictClassesFromImage(Bitmap image) {
        // Preprocessing: Resize, convert type
        ByteBuffer[] inputs = preprocess(image);

        // Inference
        tfLiteInterpreter.runForMultipleInputsOutputs(inputs, new HashMap<>());

        // Postprocessing: Compute top K indices and convert to labels
        return postprocess();
    }

    /**
     * Return the indices of the top K elements in a float buffer.
     *
     * @param fb The float buffer to read values from.
     * @param k The number of indices to return.
     * @return The indices of the top K elements in the buffer.
     */
    private static List<Integer> findTopKFloatIndices(FloatBuffer fb, int k) {
        class ValueAndIdx implements Comparable<ValueAndIdx>{
            public float value;
            public int idx;

            @Override public int compareTo(ValueAndIdx other) {
                return Float.compare(value, other.value);
            }

            public ValueAndIdx(float value, int idx) {
                this.value = value;
                this.idx = idx;
            }
        }

        PriorityQueue<ValueAndIdx> maxHeap = new PriorityQueue<>();
        int i = 0;
        while (fb.hasRemaining()) {
            maxHeap.add(new ValueAndIdx(fb.get(), i));
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
            i++;
        }

        ArrayList<Integer> topKList = maxHeap.stream().map(x -> x.idx).collect(Collectors.toCollection(ArrayList::new));
        Collections.reverse(topKList);
        return topKList;
    }

    /**
     * Return the indices of the top K elements in a byte buffer.
     *
     * @param bb The byte buffer to read values from.
     * @param k The number of indices to return.
     * @return The indices of the top K elements in the buffer.
     */
    private static List<Integer> findTopKByteIndices(ByteBuffer bb, int k) {
        class ValueAndIdx implements Comparable<ValueAndIdx>{
            public byte value;
            public int idx;

            @Override public int compareTo(ValueAndIdx other) {
                return Byte.compare(value, other.value);
            }

            public ValueAndIdx(byte value, int idx) {
                this.value = value;
                this.idx = idx;
            }
        }

        PriorityQueue<ValueAndIdx> maxHeap = new PriorityQueue<>();
        int i = 0;
        while (bb.hasRemaining()) {
            maxHeap.add(new ValueAndIdx(bb.get(), i));
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
            i++;
        }

        ArrayList<Integer> topKList = maxHeap.stream().map(x -> x.idx).collect(Collectors.toCollection(ArrayList::new));
        Collections.reverse(topKList);
        return topKList;
    }
}
