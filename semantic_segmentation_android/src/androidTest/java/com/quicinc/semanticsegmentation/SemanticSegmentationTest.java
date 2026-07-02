// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.quicinc.tflite.AIHubDefaults;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.InputStream;

@RunWith(AndroidJUnit4.class)
public class SemanticSegmentationTest {

    private static final int WARMUP_FRAMES = 3;
    private static final float MIN_FPS = 5.0f;
    private static final long MAX_INFER_MS = 50;

    private SemanticSegmentation segmentor;

    @Before
    public void setUp() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        segmentor = new SemanticSegmentation(
                context,
                context.getString(R.string.tfLiteModelAsset),
                AIHubDefaults.delegatePriorityOrder);
    }

    @After
    public void tearDown() throws Exception {
        if (segmentor != null) segmentor.close();
    }

    private Bitmap loadTestImage() throws Exception {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        try (InputStream is = testContext.getAssets().open("Sample1.jpg")) {
            Bitmap bmp = BitmapFactory.decodeStream(is);
            assertNotNull("Failed to load test image", bmp);
            return bmp;
        }
    }

    @Test
    public void testPredictReturnsValidBitmap() throws Exception {
        Bitmap output = segmentor.predict(loadTestImage(), 0);

        assertNotNull("predict() returned null", output);
        assertFalse("output bitmap is recycled", output.isRecycled());
        assertTrue("output width > 0", output.getWidth() > 0);
        assertTrue("output height > 0", output.getHeight() > 0);
    }

    @Test
    public void testInferenceTimeAndFps() throws Exception {
        Bitmap input = loadTestImage();

        // Warm up — first few frames may be slower due to delegate initialization
        for (int i = 0; i < WARMUP_FRAMES; i++) {
            segmentor.predict(input, 0);
        }

        // Measure over several frames
        int frames = 10;
        long startMs = System.currentTimeMillis();
        long totalInferNs = 0;
        for (int i = 0; i < frames; i++) {
            segmentor.predict(input, 0);
            totalInferNs += segmentor.getLastInferenceTime();
        }
        long elapsedMs = System.currentTimeMillis() - startMs;

        float fps = frames * 1000.0f / elapsedMs;
        long avgInferMs = totalInferNs / frames / 1_000_000;

        assertTrue(
            "FPS too low: " + fps + " (expected >= " + MIN_FPS + ")",
            fps >= MIN_FPS);
        assertTrue(
            "Inference too slow: " + avgInferMs + "ms (expected <= " + MAX_INFER_MS + "ms)",
            avgInferMs <= MAX_INFER_MS);
    }
}
