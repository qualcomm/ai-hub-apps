// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;
import java.util.concurrent.locks.ReentrantLock;

/**
 * FragmentRender draws the final prediction image and overlays debugging text.
 */

public class FragmentRender extends View {
    private final ReentrantLock mLock = new ReentrantLock();
    private Bitmap mBitmap = null;
    private final Rect mTargetRect = new Rect();
    private float fps;
    private long inferTime = 0;
    private long preprocessTime = 0;
    private long postprocessTime = 0;
    private final Paint mTextColor = new Paint();


    public FragmentRender(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        mTextColor.setColor(Color.WHITE);
        mTextColor.setTypeface(Typeface.DEFAULT_BOLD);
        mTextColor.setStyle(Paint.Style.FILL);
        mTextColor.setTextSize(50);
    }

    public void render(Bitmap image, float fps, long inferTime, long preprocessTime, long postprocessTime)
    {
        this.mBitmap = image;
        this.fps = fps;
        this.inferTime = inferTime;
        this.preprocessTime = preprocessTime;
        this.postprocessTime = postprocessTime;
        postInvalidate();
    }

    @SuppressLint("DefaultLocale")
    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        mLock.lock();

        if (mBitmap != null) {
            int insetHeight, insetWidth;

            float canvasRatio = (float) getWidth() / (float) getHeight();
            float bitmapRatio = (float) mBitmap.getWidth() / mBitmap.getHeight();
            if (canvasRatio > bitmapRatio) {
                insetHeight = getHeight();
                insetWidth = (int) ((float) getHeight() * bitmapRatio);
            } else {
                insetWidth = getWidth();
                insetHeight = (int) ((float) getWidth() / bitmapRatio);
            }

            int offsetWidth = (getWidth() - insetWidth) / 2;
            int offsetHeight = (getHeight() - insetHeight) / 2;
            mTargetRect.left = offsetWidth;
            mTargetRect.top = offsetHeight;
            mTargetRect.right = offsetWidth + insetWidth;
            mTargetRect.bottom = offsetHeight + insetHeight;
            canvas.drawBitmap(mBitmap, null, mTargetRect, null);
            canvas.rotate(90, 0, 0);
            canvas.translate(offsetHeight, -insetWidth - offsetWidth);

            canvas.drawText("FPS: " + String.format("%.0f", fps), 15, 50, mTextColor);
            canvas.drawText("Preprocess: " + String.format("%.0f", (float)preprocessTime / 1_000_000) + "ms", 15, 55 + 60 * 2, mTextColor);
            canvas.drawText("Infer: " + String.format("%.0f", (float)inferTime / 1_000_000) + "ms", 15, 55 + 60 * 3, mTextColor);
            canvas.drawText("Postprocess: " + String.format("%.0f", (float)postprocessTime / 1_000_000) + "ms", 15, 55 + 60 * 4, mTextColor);
            canvas.drawText("Note: Will only produce sensible results on street scenes", 15, insetWidth - 15  , mTextColor);
        }
        mLock.unlock();
    }
}
