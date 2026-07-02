// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.imageclassification;

import static org.junit.Assert.fail;

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

import java.util.Arrays;

@RunWith(AndroidJUnit4.class)
public class ImageClassificationTest {

    private static final long LAUNCH_TIMEOUT_MS = 10_000;
    private static final long INFERENCE_TIMEOUT_MS = 60_000;
    private static final String PACKAGE = "com.quicinc.imageclassification";

    private UiDevice device;

    @Before
    public void setUp() throws Exception {
        device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation());
        // Wake the device and dismiss keyguard so the UI is interactable
        device.wakeUp();
        device.executeShellCommand("wm dismiss-keyguard");
        Context context = ApplicationProvider.getApplicationContext();
        Intent intent = context.getPackageManager().getLaunchIntentForPackage(PACKAGE);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK);
        context.startActivity(intent);
        device.wait(Until.hasObject(By.pkg(PACKAGE).depth(0)), LAUNCH_TIMEOUT_MS);
    }

    private String runInferenceOnSample(String sampleName) {
        device.findObject(By.res(PACKAGE, "imageSelector")).click();
        device.wait(Until.findObject(By.text(sampleName)), LAUNCH_TIMEOUT_MS).click();

        UiObject2 runButton = device.wait(
                Until.findObject(By.res(PACKAGE, "runModelButton").enabled(true)),
                LAUNCH_TIMEOUT_MS);
        runButton.click();

        device.wait(
                Until.gone(By.res(PACKAGE, "predictionResultText").text("Inferencing...")),
                INFERENCE_TIMEOUT_MS);

        return device.findObject(By.res(PACKAGE, "predictionResultText")).getText();
    }

    private void assertContainsAny(String result, String... keywords) {
        String lower = result.toLowerCase();
        for (String kw : keywords) {
            if (lower.contains(kw.toLowerCase())) return;
        }
        fail("Expected one of " + Arrays.toString(keywords) + " in: \"" + result + "\"");
    }

    @Test
    public void testSample1_defaultDelegate() {
        String result = runInferenceOnSample("Sample1.png");
        assertContainsAny(result, "keyboard");
    }

    @Test
    public void testSample2_defaultDelegate() {
        String result = runInferenceOnSample("Sample2.png");
        assertContainsAny(result, "teddy", "toy", "bear");
    }

    @Test
    public void testSample3_defaultDelegate() {
        String result = runInferenceOnSample("Sample3.png");
        assertContainsAny(result, "mug", "cup");
    }

    @Test
    public void testSample1_cpuOnly() {
        device.findObject(By.res(PACKAGE, "cpuOnlyRadio")).click();
        String result = runInferenceOnSample("Sample1.png");
        assertContainsAny(result, "keyboard");
    }

    @Test
    public void testSample2_cpuOnly() {
        device.findObject(By.res(PACKAGE, "cpuOnlyRadio")).click();
        String result = runInferenceOnSample("Sample2.png");
        assertContainsAny(result, "teddy", "toy", "bear");
    }

    @Test
    public void testSample3_cpuOnly() {
        device.findObject(By.res(PACKAGE, "cpuOnlyRadio")).click();
        String result = runInferenceOnSample("Sample3.png");
        assertContainsAny(result, "mug", "cup");
    }
}
