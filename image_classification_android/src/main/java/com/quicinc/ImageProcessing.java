// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.RectF;

public class ImageProcessing {
    /**
     * Resize a bitmap while respecting its aspect ratio.
     * If the output image cannot fit perfectly within the requested output height / width,
     * padding is added such that the output bitmap will be the requested size.
     *
     * @param image  Image to resize
     * @param outputBitmapWidth  Final width
     * @param outputBitmapHeight  Final height
     * @param paddingValue  Value to use for padding (usually 0 or 0xFF)
     * @return Resized & padded bitmap
     */
    public static Bitmap resizeAndPadMaintainAspectRatio(
            Bitmap image,
            int outputBitmapWidth,
            int outputBitmapHeight,
            int paddingValue) {
        int width = image.getWidth();
        int height = image.getHeight();
        float ratioBitmap = (float) width / (float) height;
        float ratioMax = (float) outputBitmapWidth / (float) outputBitmapHeight;

        int finalWidth = outputBitmapWidth;
        int finalHeight = outputBitmapHeight;
        if (ratioMax > ratioBitmap) {
            finalWidth = (int) ((float)outputBitmapHeight * ratioBitmap);
        } else {
            finalHeight = (int) ((float)outputBitmapWidth / ratioBitmap);
        }

        Bitmap outputImage = Bitmap.createBitmap(outputBitmapWidth, outputBitmapHeight, Bitmap.Config.ARGB_8888);
        Canvas can = new Canvas(outputImage);
        can.drawARGB(0xFF, paddingValue, paddingValue, paddingValue);
        int left = (outputBitmapWidth - finalWidth) / 2;
        int top = (outputBitmapHeight - finalHeight) / 2;
        can.drawBitmap(image, null, new RectF(left, top, finalWidth + left, finalHeight + top), null);
        return outputImage;
    }
}
