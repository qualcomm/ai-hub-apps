// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.superresolution;

import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;

import android.content.Context;
import android.content.Intent;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiDevice;
import androidx.test.uiautomator.UiObject2;
import androidx.test.uiautomator.Until;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class SuperResolutionTest {

    private static final long LAUNCH_TIMEOUT_MS = 10_000;
    private static final long INFERENCE_TIMEOUT_MS = 60_000;
    private static final String PACKAGE = "com.quicinc.superresolution";

    private UiDevice device;

    @Before
    public void setUp() throws Exception {
        device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation());
        device.wakeUp();
        device.executeShellCommand("wm dismiss-keyguard");
        Context context = ApplicationProvider.getApplicationContext();
        Intent intent = context.getPackageManager().getLaunchIntentForPackage(PACKAGE);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK);
        context.startActivity(intent);
        device.wait(Until.hasObject(By.pkg(PACKAGE).depth(0)), LAUNCH_TIMEOUT_MS);
    }

    private void runInferenceOnSample(String sampleName) {
        device.findObject(By.res(PACKAGE, "imageSelector")).click();
        device.wait(Until.findObject(By.text(sampleName)), LAUNCH_TIMEOUT_MS).click();

        UiObject2 runButton = device.wait(
                Until.findObject(By.res(PACKAGE, "runModelButton").enabled(true)),
                LAUNCH_TIMEOUT_MS);
        runButton.click();

        device.wait(
                Until.gone(By.res(PACKAGE, "inferenceTimeResultText").text("-- ms")),
                INFERENCE_TIMEOUT_MS);
    }

    private void assertTimingNonEmpty(String viewId) {
        UiObject2 view = device.findObject(By.res(PACKAGE, viewId));
        assertNotNull("View not found: " + viewId, view);
        String text = view.getText();
        assertNotNull("Timing text is null", text);
        assertNotEquals("Expected timing result in " + viewId + ", got: " + text, "-- ms", text);
    }

    @Test
    public void testSample1_defaultDelegate() {
        runInferenceOnSample("Sample1.jpg");
        assertTimingNonEmpty("inferenceTimeResultText");
        assertTimingNonEmpty("predictionTimeResultText");
    }

    @Test
    public void testSample2_defaultDelegate() {
        runInferenceOnSample("Sample2.jpg");
        assertTimingNonEmpty("inferenceTimeResultText");
        assertTimingNonEmpty("predictionTimeResultText");
    }

    @Test
    public void testSample1_cpuOnly() {
        device.findObject(By.res(PACKAGE, "cpuOnlyRadio")).click();
        runInferenceOnSample("Sample1.jpg");
        assertTimingNonEmpty("inferenceTimeResultText");
        assertTimingNonEmpty("predictionTimeResultText");
    }
}
