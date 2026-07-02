// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;

import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.osgi.OpenCVNativeLoader;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map;

public class SemanticSegmentation implements AutoCloseable {
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final int[] inputShape;
    private final int[] outputShape;
    private long preprocessingTime;
    private long inferenceTime;
    private long postprocessingTime;
    private final int NUM_CLASSES = 19; // Output classes (CityScapes dataset)
    // Re-usable memory
    private final ByteBuffer inputByteBuffer;
    private final float[] inputFloatArray;
    private final Mat inputMatAbgr;
    private final Mat inputMatBgr;
    private final Mat inputMatRgb;
    private Mat outputCategories;

    /**
     * Create an Semantic Segmentor from the given model.
     * Uses default compute units: NPU, GPU, CPU.
     * Ignores compute units that fail to load.
     *
     * @param context App context.
     * @param modelPath Model path to load.
     * @throws IOException If the model can't be read from disk.
     */
    public SemanticSegmentation(Context context,
                               String modelPath,
                               TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Initialize OpenCV
        new OpenCVNativeLoader().init();

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
        DataType inputType = inputTensor.dataType();
        assert inputShape.length == 4; // 4D Input Tensor: [Batch, Input Height, Input Width, Color Channels]
        assert inputShape[0] == 1; // Batch size is 1
        assert inputShape[3] == 3; // Input tensor should have 3 channels
        assert inputType == DataType.FLOAT32; // Requires an unquantized FFNet variant

        assert tfLiteInterpreter.getOutputTensorCount() == 1;
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        outputShape = outputTensor.shape();
        DataType outputType = outputTensor.dataType();
        assert outputShape.length == 4; // 4D Output Tensor: [Batch, Output Height, Output Width, Classes]
        assert outputShape[0] == 1; // Batch size is 1
        assert outputShape[3] == NUM_CLASSES;
        assert outputType == DataType.FLOAT32; // Requires an unquantized FFNet variant

        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        int outputHeight = outputShape[1];
        int outputWidth = outputShape[2];

        // Allocate re-usable memory
        inputByteBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * 3 * 4);
        inputByteBuffer.order(ByteOrder.nativeOrder());

        inputFloatArray = new float[inputHeight * inputWidth * 3];

        inputMatAbgr = new Mat(inputWidth, inputHeight, CvType.CV_8UC4);
        inputMatRgb = new Mat(inputWidth, inputHeight, CvType.CV_8UC3);
        inputMatBgr = new Mat(inputWidth, inputHeight, CvType.CV_8UC3);
        outputCategories = new Mat(outputWidth, outputHeight, CvType.CV_32FC1);
    }

    /**
     * @return neural network model input width
     */
    public int getInputWidth() {
        return inputShape[2];
    }

    /**
     * @return neural network model input height
     */
    public int getInputHeight() {
        return inputShape[1];
    }

    /**
     * Free resources used by the segmentor.
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
        return preprocessingTime;
    }

    /**
     * @return last inference time in microseconds.
     */
    public long getLastInferenceTime() {
        return inferenceTime;
    }

    /**
     * @return last postprocessing time in microseconds.
     */
    public long getLastPostprocessingTime() {
        return postprocessingTime;
    }

    /**
     * Predicts and overlays
     * @param image Input image
     * @param sensorOrientation Sensor orientation in degrees. If input image is rotated this
     *                          number of degrees clockwise, the image shoudl be upright.
     *
     * @return RGB bitmap of same size and orientation as input image, but with predictions overlay.
     */
    public Bitmap predict(Bitmap image, int sensorOrientation) {
        // The most common sensor orientation is 90, so we will use it for shape examples.
        // Image comes in requiring 90 degrees cw rotation to be correct;
        // its size is then 1024 x 2048 (width x height)

        long preStartTime = System.nanoTime();

        //
        // Preprocessing
        //

        // Copy input image into OpenCV Mat
        Utils.bitmapToMat(image, inputMatAbgr);

        // OpenCV loads the image as BGR, but the network expects RGB
        Imgproc.cvtColor(inputMatAbgr, inputMatRgb, Imgproc.COLOR_BGRA2RGB);
        Imgproc.cvtColor(inputMatAbgr, inputMatBgr, Imgproc.COLOR_BGRA2BGR);

        // Rotate if necessary
        Mat correctRotInputImageRgb = new Mat();
        switch (sensorOrientation) {
            case 0:
                correctRotInputImageRgb = inputMatRgb;
                break;
            case 90:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_90_COUNTERCLOCKWISE);
                break;
            case 180:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_180);
                break;
            case 270:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_90_CLOCKWISE);
                break;
            default:
                break;
        }

        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        // Scale image to the network
        Mat scaledImage = new Mat(inputHeight, inputWidth, CvType.CV_8UC3);
        Imgproc.resize(correctRotInputImageRgb, scaledImage, scaledImage.size(), 0, 0, Imgproc.INTER_LINEAR);
        scaledImage.convertTo(scaledImage, CvType.CV_32FC3, 1 / 255f);

        //
        // TFLite inference
        //

        // Convert from OpenCV to TFLite expected ByteBuffer
        // To minimize IO overhead, we create a direct-allocated buffer in native order.

        scaledImage.get(0, 0, inputFloatArray);
        FloatBuffer inputFloatBuffer = inputByteBuffer.asFloatBuffer();
        inputFloatBuffer.put(inputFloatArray);

        long inferenceStartTime = System.nanoTime();
        preprocessingTime = inferenceStartTime - preStartTime;

        // Run inference
        ByteBuffer[] inputs = new ByteBuffer[] {inputByteBuffer};
        tfLiteInterpreter.runForMultipleInputsOutputs(inputs, new HashMap<>());

        //
        // Postprocessing
        //

        long postStartTime = System.nanoTime();
        inferenceTime = postStartTime - inferenceStartTime;

        int outputHeight = outputShape[1];
        int outputWidth = outputShape[2];

        // Convert output to 3D OpenCV image
        Mat outputs = new Mat(new int[]{outputHeight, outputWidth, NUM_CLASSES}, CvType.CV_32F);
        ByteBuffer outputBuffer = tfLiteInterpreter.getOutputTensor(0).asReadOnlyBuffer();
        FloatBuffer floatBuf = outputBuffer.asFloatBuffer();
        float[] arr = new float[floatBuf.remaining()];
        floatBuf.get(arr);
        outputs.put(new int[]{0, 0, 0}, arr);

        // Take argmax (top class prediction) and scale up
        Core.reduceArgMax(outputs, outputCategories, 2);
        outputCategories.convertTo(outputCategories, CvType.CV_8UC1);
        outputCategories = outputCategories.reshape(1, new int[]{outputHeight, outputWidth});

        // Spread out the indices to fill up [0, 1]
        // This will make better use of the rainbow color map below
        Core.multiply(outputCategories, new Scalar(255.0f / (float)(NUM_CLASSES - 1)), outputCategories);

        // Rotate output to match input
        switch (sensorOrientation) {
            case 90:
                Core.rotate(outputCategories, outputCategories, Core.ROTATE_90_CLOCKWISE);
                break;
            case 180:
                Core.rotate(outputCategories, outputCategories, Core.ROTATE_180);
                break;
            case 270:
                Core.rotate(outputCategories, outputCategories, Core.ROTATE_90_COUNTERCLOCKWISE);
                break;
            default:
                break;
        }

        // Resize back to input size
        Mat resizeImage = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC1);
        Imgproc.resize(outputCategories, resizeImage, resizeImage.size(), 0, 0, Imgproc.INTER_LINEAR);

        // Convert grayscale indices map to an RGB image (with rainbow color map)
        Mat mask = new Mat(image.getWidth(), image.getHeight(), CvType.CV_8UC3);
        Imgproc.applyColorMap(resizeImage, mask, Imgproc.COLORMAP_RAINBOW);

        // Overlay on top of camera preview
        Core.addWeighted(inputMatBgr, 0.7, mask, 0.3, 0.0, inputMatBgr);

        // Convert from OpenCV Mat to Android Bitmap
        Bitmap outputBitmap = Bitmap.createBitmap(inputMatBgr.width(), inputMatBgr.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(inputMatBgr, outputBitmap);
        long endTime = System.nanoTime();
        postprocessingTime = endTime - postStartTime;
        return outputBitmap;
    }
}
