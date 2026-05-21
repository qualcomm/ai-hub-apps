// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.chatapp;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
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

@RunWith(AndroidJUnit4.class)
public class ChatAppTest {

    private static final long LAUNCH_TIMEOUT_MS = 10_000;
    private static final long RESPONSE_WAIT_MS = 15_000;
    private static final String PACKAGE = "com.quicinc.chatapp";
    private static final String TEST_PROMPT = "What is gravity? Keep the answer under 50 words.";

    // Performance thresholds
    private static final long MAX_TTFT_MS = 5_000;  // TTFT must be under 5 seconds
    private static final double MIN_TPS = 5.0;       // Must produce at least 5 tokens/sec

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

    @Test
    public void testSendPromptAndReceiveResponse() throws Exception {
        device.wait(Until.findObject(By.res(PACKAGE, "llm")), LAUNCH_TIMEOUT_MS).click();

        device.wait(Until.findObject(By.res(PACKAGE, "user_input")), LAUNCH_TIMEOUT_MS)
              .setText(TEST_PROMPT);
        device.findObject(By.res(PACKAGE, "send_button")).click();

        // Wait for stats bar to appear — signals inference is complete
        // Format: "TTFT: 1234 ms  |  TPS: 23.4 tok/s"
        UiObject2 statsBar = device.wait(
                Until.findObject(By.res(PACKAGE, "stats_bar").textContains("TTFT:")),
                RESPONSE_WAIT_MS);
        assertNotNull("Stats bar did not appear — inference may have timed out", statsBar);

        // Verify response is non-empty (welcome message is index 0, LLM response is last)
        java.util.List<UiObject2> botMessages = device.findObjects(By.res(PACKAGE, "bot_message"));
        assertTrue("Expected at least 2 bot messages (welcome + response), found: " + botMessages.size(),
                botMessages.size() >= 2);
        String responseText = botMessages.get(botMessages.size() - 1).getText();
        assertNotNull("LLM response text is null", responseText);
        assertFalse("LLM response is empty", responseText.trim().isEmpty());

        // Verify performance metrics
        String stats = statsBar.getText();
        try {
            java.util.regex.Matcher ttftMatcher = java.util.regex.Pattern
                    .compile("TTFT:\\s*(\\d+)\\s*ms")
                    .matcher(stats);
            assertTrue("Could not parse TTFT from: " + stats, ttftMatcher.find());
            long ttftMs = Long.parseLong(ttftMatcher.group(1));

            java.util.regex.Matcher tpsMatcher = java.util.regex.Pattern
                    .compile("TPS:\\s*([\\d.]+)\\s*tok/s")
                    .matcher(stats);
            assertTrue("Could not parse TPS from: " + stats, tpsMatcher.find());
            double tps = Double.parseDouble(tpsMatcher.group(1));

            assertTrue("TTFT too high: " + ttftMs + "ms (max " + MAX_TTFT_MS + "ms)",
                    ttftMs <= MAX_TTFT_MS);
            assertTrue("TPS too low: " + tps + " tok/s (min " + MIN_TPS + " tok/s)",
                    tps >= MIN_TPS);
        } catch (NumberFormatException e) {
            fail("Failed to parse performance metrics from stats bar: " + stats);
        }
    }
}
