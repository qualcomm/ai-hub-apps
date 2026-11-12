// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo

import com.geniex.sdk.bean.GenerationConfig

// Configuration sample for generation with defaults compatible with bridge.
// maxTokens must be set explicitly: geniex maps 0 to 128 (llama_cpp LLM) or
// 512 (VLM), not "no limit".
data class GenerationConfigSample(
    var maxTokens: Int = 2048,
    var stopWords: List<String>? = null,
    var stopCount: Int = 0,
    var imagePaths: List<String>? = null,
    var imageCount: Int = 0,
    var audioPaths: List<String>? = null,
    var audioCount: Int = 0,
) {
    // Convert to GenerationConfig with minimal sampler setup for bridge compatibility
    // Sampler config uses bridge defaults (no custom parameters applied)
    fun toGenerationConfig(): GenerationConfig =
        GenerationConfig(
            maxTokens = this.maxTokens,
            stopWords = this.stopWords?.toTypedArray(),
            stopCount = this.stopCount,
            imagePaths = this.imagePaths?.toTypedArray(),
            imageCount = this.imageCount,
            audioPaths = this.audioPaths?.toTypedArray(),
            audioCount = this.audioCount,
        )
}
