// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo

import android.content.Context
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.StaleObjectException
import androidx.test.uiautomator.Until
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device test for the GenieX Demo chat flow.
 *
 * The GenieX demo downloads its model at runtime, so this test drives the full
 * path on a real device: select the smallest catalog model, download it, load
 * it onto each available compute unit (NPU, GPU, CPU), send a prompt, and
 * assert a non-empty assistant response renders for each. Running on every
 * compute unit catches a model that loads on one but fails on another (e.g.
 * NPU).
 *
 * The test asserts on response presence and non-emptiness. The device must
 * have network egress to download the model; the smallest catalog entry
 * (Qwen3-0.6B GGUF, ~0.4 GB) is used to keep the download within the timeouts
 * below.
 */
@RunWith(AndroidJUnit4::class)
class GenieXChatTest {

    private val device: UiDevice =
        UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())

    @Before
    fun setUp() {
        device.wakeUp()
        device.executeShellCommand("wm dismiss-keyguard")

        val context = ApplicationProvider.getApplicationContext<Context>()
        val intent = context.packageManager.getLaunchIntentForPackage(PACKAGE)!!
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK)
        context.startActivity(intent)
        assertTrue(
            "App did not launch",
            device.wait(Until.hasObject(By.pkg(PACKAGE).depth(0)), LAUNCH_TIMEOUT_MS)
        )
    }

    @Test
    fun selectModelDownloadLoadAndChat() {
        // 1. Select the target model. It is the first entry in
        //    model_list.json, so the Spinner auto-selects it at startup and
        //    fires onItemSelected — no dropdown interaction is needed in the
        //    common case. We only open the dropdown if the displayed selection
        //    is not already the target (e.g. if the catalog is reordered).
        val spinner = device.wait(Until.findObject(By.res(PACKAGE, "sp_model_list")), LAUNCH_TIMEOUT_MS)
        assertNotNull("Model spinner not found", spinner)
        // The selected model's displayName is rendered in a tv_model_id child.
        // If it is not already showing the target, open the dropdown and pick it.
        if (!device.hasObject(By.text(MODEL_DISPLAY_NAME))) {
            selectModelFromDropdown()
        }
        // Confirm the intended model is the active selection before proceeding.
        assertTrue(
            "Model '$MODEL_DISPLAY_NAME' is not the active selection",
            device.wait(Until.hasObject(By.text(MODEL_DISPLAY_NAME)), UI_TIMEOUT_MS)
        )

        // 2. Download the model. The download panel (ll_downloading) appears
        //    while the pull runs and is hidden on completion. Wait for it to
        //    appear, then to disappear, with a generous network timeout.
        device.wait(Until.findObject(By.res(PACKAGE, "btn_download")), UI_TIMEOUT_MS).click()
        // It may already be cached from a prior run — only wait for the panel
        // to clear if it actually showed up.
        device.wait(Until.hasObject(By.res(PACKAGE, "ll_downloading")), UI_TIMEOUT_MS)
        assertTrue(
            "Model download did not complete within ${DOWNLOAD_TIMEOUT_MS / 1000}s",
            device.wait(Until.gone(By.res(PACKAGE, "ll_downloading")), DOWNLOAD_TIMEOUT_MS)
        )

        // 3. Load and chat on each compute unit the model offers. A GGUF model
        //    exposes NPU, GPU and CPU; exercising all three catches a model that
        //    loads on one unit but fails on another. NPU runs first since it is
        //    the most likely to surface a problem.
        for (unit in COMPUTE_UNITS) {
            loadChatAndAssertOnUnit(unit)
            unloadModel()
        }
    }

    /**
     * Loads the selected model onto [unit], sends a prompt, and asserts a
     * non-empty assistant response renders. Assumes the model is downloaded and
     * no model is currently loaded.
     */
    private fun loadChatAndAssertOnUnit(unit: ComputeUnit) {
        device.wait(Until.findObject(By.res(PACKAGE, "btn_load_model")), UI_TIMEOUT_MS).click()

        // The compute-unit picker appears for multi-unit (GGUF) models. Select
        // the radio for this unit and confirm. GPU reveals an n-layers field
        // that defaults to 999, so no extra input is needed.
        val radio = device.wait(Until.findObject(By.res(PACKAGE, unit.radioId)), UI_TIMEOUT_MS)
        assertNotNull("Compute-unit radio '${unit.radioId}' not found in load dialog", radio)
        radio.click()
        val okButton = device.wait(Until.findObject(By.text("OK")), UI_TIMEOUT_MS)
        assertNotNull("Load dialog OK button not found", okButton)
        okButton.click()

        // Load success reveals the Unload button (gone until a model loads).
        assertTrue(
            "Model did not load on ${unit.label} within ${LOAD_TIMEOUT_MS / 1000}s",
            device.wait(Until.hasObject(By.res(PACKAGE, "btn_unload_model")), LOAD_TIMEOUT_MS)
        )

        // Send a prompt.
        val input = device.wait(Until.findObject(By.res(PACKAGE, "et_input")), UI_TIMEOUT_MS)
        assertNotNull("Input field not found", input)
        input.text = TEST_PROMPT
        device.wait(Until.findObject(By.res(PACKAGE, "btn_send")), UI_TIMEOUT_MS).click()

        // Wait for inference and assert a non-empty assistant response. After
        // sending, the chat list holds [USER prompt, ASSISTANT reply], both
        // rendered in `tv_message`; the assistant reply is the last one and must
        // differ from the prompt and be non-empty.
        var responseText: String? = null
        var waited = 0L
        while (waited < RESPONSE_TIMEOUT_MS) {
            val messages: List<UiObject2> = device.findObjects(By.res(PACKAGE, "tv_message"))
            if (messages.isNotEmpty()) {
                val last = messages.last().text
                if (!last.isNullOrBlank() && last != TEST_PROMPT) {
                    responseText = last
                    break
                }
            }
            device.waitForIdle(POLL_INTERVAL_MS)
            waited += POLL_INTERVAL_MS
        }
        assertNotNull("No assistant response on ${unit.label} within ${RESPONSE_TIMEOUT_MS / 1000}s", responseText)
        assertFalse("Assistant response on ${unit.label} is empty", responseText!!.trim().isEmpty())
    }

    /** Unloads the current model and waits for the Unload button to disappear. */
    private fun unloadModel() {
        device.wait(Until.findObject(By.res(PACKAGE, "btn_unload_model")), UI_TIMEOUT_MS).click()
        assertTrue(
            "Model did not unload within ${UI_TIMEOUT_MS / 1000}s",
            device.wait(Until.gone(By.res(PACKAGE, "btn_unload_model")), UI_TIMEOUT_MS)
        )
    }

    /**
     * Opens the model Spinner and taps the target entry. The dropdown animates,
     * so the matched node can go stale between match and click — retry a few
     * times to absorb that.
     */
    private fun selectModelFromDropdown() {
        repeat(DROPDOWN_RETRIES) { attempt ->
            try {
                device.wait(Until.findObject(By.res(PACKAGE, "sp_model_list")), UI_TIMEOUT_MS).click()
                val option = device.wait(Until.findObject(By.text(MODEL_DISPLAY_NAME)), UI_TIMEOUT_MS)
                assertNotNull("Model '$MODEL_DISPLAY_NAME' not in catalog dropdown", option)
                option.click()
                return
            } catch (e: StaleObjectException) {
                if (attempt == DROPDOWN_RETRIES - 1) throw e
                device.waitForIdle(POLL_INTERVAL_MS)
            }
        }
    }

    /** A compute unit offered in the load dialog, identified by its radio id. */
    private data class ComputeUnit(val label: String, val radioId: String)

    companion object {
        private const val PACKAGE = "com.geniex.demo"

        // Smallest entry in src/main/assets/model_list.json — keeps the runtime
        // download small enough for CI timeouts.
        private const val MODEL_DISPLAY_NAME = "Qwen3-0.6B (GGUF)"
        private const val TEST_PROMPT = "What is gravity? Answer in under 30 words."

        // Compute units the GGUF model offers, NPU first (most likely to fail).
        private val COMPUTE_UNITS = listOf(
            ComputeUnit("NPU", "rb_npu"),
            ComputeUnit("GPU", "rb_gpu"),
            ComputeUnit("CPU", "rb_cpu"),
        )

        private const val LAUNCH_TIMEOUT_MS = 15_000L
        private const val UI_TIMEOUT_MS = 10_000L
        private const val DOWNLOAD_TIMEOUT_MS = 600_000L   // up to 10 min for the model pull
        private const val LOAD_TIMEOUT_MS = 120_000L        // model load onto a compute unit
        private const val RESPONSE_TIMEOUT_MS = 120_000L    // first non-empty generation
        private const val POLL_INTERVAL_MS = 2_000L
        private const val DROPDOWN_RETRIES = 3
    }
}
