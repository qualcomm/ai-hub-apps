// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.objectdetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;

import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.osgi.OpenCVNativeLoader;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ObjectDetection implements AutoCloseable {
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final List<String> labelList;
    private final int[] inputShape;
    private final int[] outputBoxesShape;
    private final int[] outputScoresShape;
    private final int[] outputClassIdxShape;
    private final DataType inputType;
    private final DataType outputBoxesType;
    private final DataType outputScoresType;
    private final Tensor.QuantizationParams inputQuantParams;
    private final Tensor.QuantizationParams outputBoxesQuantParams;
    private final Tensor.QuantizationParams outputScoresQuantParams;
    private final boolean outputClassIs32bit;
    private final int numBoxes;
    private long preprocessingTime;
    private long inferenceTime;
    private long postprocessingTime;
    // Re-usable memory
    private final ByteBuffer inputByteBuffer;
    private final float[] inputFloatArray;
    private final Mat inputMatAbgr;
    private final Mat inputMatRgb;

    private static final float INVALID_ANCHOR = -10000.0f;

    /**
     * Create an Object Detector from the given model. Uses default compute units: NPU, GPU, CPU.
     * Ignores compute units that fail to load.
     *
     * @param context App context.
     * @param modelPath Model path to load.
     * @throws IOException If the model can't be read from disk.
     */
    public ObjectDetection(
            Context context,
            String modelPath,
            String labelsPath,
            TFLiteHelpers.DelegateType[][] delegatePriorityOrder)
            throws IOException, NoSuchAlgorithmException {
        // Initialize OpenCV
        new OpenCVNativeLoader().init();

        // Load labels
        try (BufferedReader labelsFile =
                new BufferedReader(new InputStreamReader(context.getAssets().open(labelsPath)))) {
            labelList = labelsFile.lines().collect(Collectors.toCollection(ArrayList::new));
        }

        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash =
                TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult =
                TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                        modelAndHash.first,
                        delegatePriorityOrder,
                        AIHubDefaults.numCPUThreads,
                        context.getApplicationInfo().nativeLibraryDir,
                        context.getCacheDir().getAbsolutePath(),
                        modelAndHash.second);
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;

        // Validate TF Lite model fits requirements for this app
        assert tfLiteInterpreter.getInputTensorCount() == 1;

        int batchSize = 1;

        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        inputShape = inputTensor.shape();
        inputType = inputTensor.dataType();
        assert inputShape.length
                == 4; // 4D Input Tensor: [Batch, Input Height, Input Width, Color Channels]
        assert inputShape[0] == batchSize;
        assert inputShape[3] == 3; // Input tensor should have 3 channels
        assert inputType == DataType.FLOAT32
                || inputType == DataType.UINT8; // FP32 and UINT8 (quantized) input supported

        assert tfLiteInterpreter.getOutputTensorCount() == 3;

        Tensor outputBoxesTensor = tfLiteInterpreter.getOutputTensor(0);
        outputBoxesShape = outputBoxesTensor.shape();
        outputBoxesType = outputBoxesTensor.dataType();
        assert outputBoxesShape.length == 3; // 3D Output Tensor: [Batch, Boxes, 4]
        assert outputBoxesShape[0] == batchSize;
        assert outputBoxesShape[2] == 4;
        numBoxes = outputBoxesShape[1];
        assert outputBoxesType == DataType.FLOAT32
                || outputBoxesType == DataType.UINT8; // FP32 and UINT8 (quantized) output supported

        Tensor outputScoresTensor = tfLiteInterpreter.getOutputTensor(1);
        outputScoresShape = outputScoresTensor.shape();
        outputScoresType = outputScoresTensor.dataType();
        assert outputScoresShape.length == 2; // 2D Output Tensor: [Batch, Scores]
        assert outputScoresShape[0] == batchSize;
        assert outputScoresType == DataType.FLOAT32
                || outputScoresType
                        == DataType.UINT8; // FP32 and UINT8 (quantized) output supported

        Tensor outputClassIdxTensor = tfLiteInterpreter.getOutputTensor(2);
        outputClassIdxShape = outputClassIdxTensor.shape();
        DataType outputClassIdxType = outputClassIdxTensor.dataType();
        assert outputClassIdxShape.length == 2; // 2D Output Tensor: [Batch, ClassIdx]
        assert outputClassIdxShape[0] == batchSize;
        assert outputClassIdxType == DataType.UINT8 || outputClassIdxType == DataType.INT32;

        outputClassIs32bit = outputClassIdxType == DataType.INT32;

        assert numBoxes == outputScoresShape[1];
        assert numBoxes == outputClassIdxShape[1];

        // Quantization params (scale + zero point); identity for float tensors.
        inputQuantParams = inputTensor.quantizationParams();
        outputBoxesQuantParams = outputBoxesTensor.quantizationParams();
        outputScoresQuantParams = outputScoresTensor.quantizationParams();

        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        // Allocate re-usable memory. Quantized models take 1 byte per element (uint8),
        // float models take 4 bytes.
        int inputBytesPerElement = inputType == DataType.FLOAT32 ? 4 : 1;
        inputByteBuffer =
                ByteBuffer.allocateDirect(inputHeight * inputWidth * 3 * inputBytesPerElement);
        inputByteBuffer.order(ByteOrder.nativeOrder());

        inputFloatArray = new float[inputHeight * inputWidth * 3];

        inputMatAbgr = new Mat(inputWidth, inputHeight, CvType.CV_8UC4);
        inputMatRgb = new Mat(inputWidth, inputHeight, CvType.CV_8UC3);
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

    /** Free resources used by the detector. */
    @Override
    public void close() {
        tfLiteInterpreter.close();
        for (Delegate delegate : tfLiteDelegateStore.values()) {
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
     * Read an output tensor as a float array, dequantizing when the tensor is uint8.
     *
     * @param outputIndex Index of the output tensor to read.
     * @param type Data type of the output tensor.
     * @param quantParams Quantization params of the output tensor (used when uint8).
     * @return The output values as float. For uint8 tensors, dequantized as scale * (q -
     *     zeroPoint).
     */
    private float[] readFloatOutput(
            int outputIndex, DataType type, Tensor.QuantizationParams quantParams) {
        ByteBuffer outputBuffer = tfLiteInterpreter.getOutputTensor(outputIndex).asReadOnlyBuffer();
        if (type == DataType.FLOAT32) {
            FloatBuffer floatBuf = outputBuffer.asFloatBuffer();
            float[] out = new float[floatBuf.remaining()];
            floatBuf.get(out);
            return out;
        }
        // Quantized uint8: dequantize as scale * (q - zeroPoint).
        float scale = quantParams.getScale();
        int zeroPoint = quantParams.getZeroPoint();
        float[] out = new float[outputBuffer.remaining()];
        for (int i = 0; i < out.length; ++i) {
            int q = outputBuffer.get(i) & 0xFF; // unsigned
            out[i] = scale * (q - zeroPoint);
        }
        return out;
    }

    /**
     * Predicts and overlays
     *
     * @param image Input image
     * @param sensorOrientation Sensor orientation in degrees. If input image is rotated this number
     *     of degrees clockwise, the image should be upright.
     * @return RGB bitmap of same size and orientation as input image, but with predictions overlay.
     */
    public void predict(Bitmap image, int sensorOrientation, ArrayList<RectangleBox> BBlist) {
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

        // Rotate if necessary
        Mat correctRotInputImageRgb = new Mat();
        switch (sensorOrientation) {
            case 0:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_90_COUNTERCLOCKWISE);
                break;
            case 90:
                correctRotInputImageRgb = inputMatRgb;
                break;
            case 180:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_90_CLOCKWISE);
                break;
            case 270:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_180);
                break;
            default:
                break;
        }

        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        // Scale image to the network
        Mat scaledImage = new Mat(inputHeight, inputWidth, CvType.CV_8UC3);
        Imgproc.resize(
                correctRotInputImageRgb,
                scaledImage,
                scaledImage.size(),
                0,
                0,
                Imgproc.INTER_LINEAR);
        scaledImage.convertTo(scaledImage, CvType.CV_32FC3, 1 / 255f);

        //
        // TFLite inference
        //

        // Convert from OpenCV to TFLite expected ByteBuffer
        // To minimize IO overhead, we create a direct-allocated buffer in native order.

        scaledImage.get(0, 0, inputFloatArray);
        inputByteBuffer.rewind();
        if (inputType == DataType.FLOAT32) {
            FloatBuffer inputFloatBuffer = inputByteBuffer.asFloatBuffer();
            inputFloatBuffer.put(inputFloatArray);
        } else {
            // Quantize normalized [0, 1] values to uint8: q = round(v / scale) + zeroPoint.
            float scale = inputQuantParams.getScale();
            int zeroPoint = inputQuantParams.getZeroPoint();
            for (float v : inputFloatArray) {
                int q = Math.round(v / scale) + zeroPoint;
                q = Math.max(0, Math.min(255, q));
                inputByteBuffer.put((byte) q);
            }
            // put() advanced the position; rewind so the interpreter reads from the start.
            inputByteBuffer.rewind();
        }

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

        // Extract outputs, dequantizing to float when the model output is uint8.
        float[] bboxes = readFloatOutput(0, outputBoxesType, outputBoxesQuantParams);
        float[] scores = readFloatOutput(1, outputScoresType, outputScoresQuantParams);

        ByteBuffer outputClassIdxBuffer = tfLiteInterpreter.getOutputTensor(2).asReadOnlyBuffer();
        byte[] classIdxByte = new byte[outputClassIdxBuffer.remaining()];
        int[] classIdx = new int[outputClassIdxShape[1]];
        if (outputClassIs32bit) {
            outputClassIdxBuffer.order(ByteOrder.nativeOrder());
            IntBuffer intBuffer = outputClassIdxBuffer.asIntBuffer();
            intBuffer.get(classIdx);
        } else {
            outputClassIdxBuffer.get(classIdxByte);
            for (int i = 0; i < classIdxByte.length; ++i) {
                classIdx[i] = (int) classIdxByte[i];
            }
        }
        float[][] updatedBoxes = new float[numBoxes][4];
        for (int i = 0; i < numBoxes; i++) {
            if (scores[i] >= 0.2) {
                float x0 = (float) bboxes[i * 4];
                float y0 = (float) bboxes[i * 4 + 1];
                float x1 = (float) bboxes[i * 4 + 2];
                float y1 = (float) bboxes[i * 4 + 3];

                switch (sensorOrientation) {
                    case 0:
                        updatedBoxes[i][0] = inputHeight - y1;
                        updatedBoxes[i][1] = x0;
                        updatedBoxes[i][2] = inputHeight - y0;
                        updatedBoxes[i][3] = x1;
                        break;
                    case 90:
                        updatedBoxes[i][0] = x0;
                        updatedBoxes[i][1] = y0;
                        updatedBoxes[i][2] = x1;
                        updatedBoxes[i][3] = y1;
                        break;
                    case 180:
                        updatedBoxes[i][0] = y0;
                        updatedBoxes[i][1] = inputWidth - x1;
                        updatedBoxes[i][2] = y1;
                        updatedBoxes[i][3] = inputWidth - x0;
                        break;
                    case 270:
                        updatedBoxes[i][0] = inputWidth - x1;
                        updatedBoxes[i][1] = inputHeight - y1;
                        updatedBoxes[i][2] = inputWidth - x0;
                        updatedBoxes[i][3] = inputHeight - y0;
                        break;
                    default:
                        break;
                }
            } else {
                scores[i] = INVALID_ANCHOR;
            }
        }

        NMS nms = new NMS();
        int[] result_indices = nms.nmsScoreFilter(updatedBoxes, scores, 20, 0.2f);

        float scaleHeight = (float) image.getHeight() / getInputHeight();
        float scaleWidth = (float) image.getWidth() / getInputWidth();

        for (int index : result_indices) {
            if (index == 0) {
                continue;
            }

            float[] temp_boxes = updatedBoxes[index];
            RectangleBox tempbox = new RectangleBox();
            tempbox.left = temp_boxes[0] * scaleWidth;
            tempbox.bottom = temp_boxes[1] * scaleHeight;
            tempbox.right = temp_boxes[2] * scaleWidth;
            tempbox.top = temp_boxes[3] * scaleHeight;
            tempbox.confidence = scores[index];
            tempbox.classIdx = (int) classIdx[index];
            tempbox.label = labelList.get((int) classIdx[index] % labelList.size());
            BBlist.add(tempbox);
        }
        long endTime = System.nanoTime();
        postprocessingTime = endTime - postStartTime;
    }

    public class NMS {
        private float computeOverlapAreaRate(float[] anchor1, float[] anchor2) {

            float xx1 = Math.max(anchor1[0], anchor2[0]);
            float yy1 = Math.max(anchor1[1], anchor2[1]);
            float xx2 = Math.min(anchor1[2], anchor2[2]);
            float yy2 = Math.min(anchor1[3], anchor2[3]);

            float w = xx2 - xx1 + 1;
            float h = yy2 - yy1 + 1;
            if (w < 0 || h < 0) {
                return 0;
            }

            float inter = w * h;

            float anchor1_area1 = (anchor1[2] - anchor1[0] + 1) * (anchor1[3] - anchor1[1] + 1);
            float anchor2_area1 = (anchor2[2] - anchor2[0] + 1) * (anchor2[3] - anchor2[1] + 1);

            return inter / (anchor1_area1 + anchor2_area1 - inter);
        }

        public int[] nmsScoreFilter(float[][] anchors, float[] score, int topN, float thresh) {
            int length = anchors.length;
            int count = 0;

            for (int i = 0; i < length; i++) {
                if (score[i] == INVALID_ANCHOR) {
                    continue;
                }
                if (++count >= topN) {
                    break;
                }
                for (int j = i + 1; j < length; j++) {
                    if (score[j] != INVALID_ANCHOR) {
                        if (computeOverlapAreaRate(anchors[i], anchors[j]) > thresh) {
                            score[j] = INVALID_ANCHOR;
                        }
                    }
                }
            }
            int outputIndex[] = new int[count];
            int j = 0;
            for (int i = 0; i < length && count > 0; i++) {
                if (score[i] != INVALID_ANCHOR) {
                    outputIndex[j++] = i;
                    count--;
                }
            }
            return outputIndex;
        }
    }
}
