// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.utils

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The vision-tower geometry an mmproj GGUF declares about itself.
 *
 * [imageSize] is the square edge length the encoder was trained at. Handing it
 * a smaller square does not save it any work, it just throws away detail, so
 * image preprocessing should target exactly this value.
 *
 * [tokenCount] is how many tokens the encoder turns that square into, which is
 * what the image costs against the model's context.
 */
data class GgufVisionConfig(
    val imageSize: Int,
    val patchSize: Int,
    val spatialMergeSize: Int,
) {
    /**
     * Patches along one edge, collapsed by the spatial merge, squared. Qwen3.5-VL
     * is 768/16 with merge 2 -> 24x24 = 576; Qwen2.5-VL is 560/14 with no merge
     * key (treated as 1) -> 40x40 = 1600.
     */
    val tokenCount: Int
        get() {
            val grid = imageSize / patchSize / spatialMergeSize
            return grid * grid
        }
}

/**
 * Minimal reader for the handful of `clip.vision.*` keys needed to size image
 * preprocessing, parsed straight from an mmproj GGUF header.
 *
 * These values differ per model — 768/16 for Qwen3.5-VL but 560/14 for
 * Qwen2.5-VL — so hardcoding one model's numbers silently mis-sizes every other
 * model in the catalog. The GenieX SDK does not surface them (there is no image
 * resolution field on GenerationConfig or ModelConfig as of geniex-android
 * 0.3.5), hence reading the file directly.
 *
 * Only the key-value block at the head of the file is read; tensor data is
 * never touched, so this stays cheap even for a multi-hundred-MB mmproj.
 */
object GgufVisionReader {
    private const val TAG = "GgufVisionReader"

    private const val GGUF_MAGIC = 0x46554747 // "GGUF" little-endian

    // Only the KV header is parsed; refuse to walk past a plausible header size.
    private const val MAX_HEADER_BYTES = 8L * 1024 * 1024

    // GGUF metadata value type tags.
    private const val TYPE_UINT8 = 0
    private const val TYPE_INT8 = 1
    private const val TYPE_UINT16 = 2
    private const val TYPE_INT16 = 3
    private const val TYPE_UINT32 = 4
    private const val TYPE_INT32 = 5
    private const val TYPE_FLOAT32 = 6
    private const val TYPE_BOOL = 7
    private const val TYPE_STRING = 8
    private const val TYPE_ARRAY = 9
    private const val TYPE_UINT64 = 10
    private const val TYPE_INT64 = 11
    private const val TYPE_FLOAT64 = 12

    private const val KEY_IMAGE_SIZE = "clip.vision.image_size"
    private const val KEY_PATCH_SIZE = "clip.vision.patch_size"
    private const val KEY_MERGE_SIZE = "clip.vision.spatial_merge_size"

    /**
     * Reads the vision geometry from [mmprojFile], or returns null if the file is
     * missing, is not a GGUF, or does not declare both an image and a patch size.
     * Callers are expected to fall back to their own default in that case —
     * failing to read metadata should degrade quality, not break inference.
     */
    fun read(mmprojFile: File): GgufVisionConfig? {
        if (!mmprojFile.isFile) {
            Log.w(TAG, "mmproj not a readable file: ${mmprojFile.absolutePath}")
            return null
        }
        return try {
            RandomAccessFile(mmprojFile, "r").use { raf ->
                parse(raf)
            }
        } catch (e: Exception) {
            // Corrupt or unexpected mmproj must not take down model loading.
            Log.w(TAG, "failed to read vision config from ${mmprojFile.name}", e)
            null
        }
    }

    private fun parse(raf: RandomAccessFile): GgufVisionConfig? {
        if (raf.length() < 24) return null

        // Header: magic, version, tensor count, kv count — all little-endian.
        if (readU32(raf).toInt() != GGUF_MAGIC) {
            Log.w(TAG, "not a GGUF file (bad magic)")
            return null
        }
        readU32(raf) // version, unused
        readU64(raf) // tensor count, unused
        val kvCount = readU64(raf)

        var imageSize: Int? = null
        var patchSize: Int? = null
        var mergeSize: Int? = null

        var i = 0L
        while (i < kvCount) {
            if (raf.filePointer > MAX_HEADER_BYTES) {
                Log.w(TAG, "gave up scanning GGUF metadata past $MAX_HEADER_BYTES bytes")
                break
            }
            val key = readString(raf) ?: break
            val value = readValue(raf, readU32(raf).toInt()) ?: break
            when (key) {
                KEY_IMAGE_SIZE -> imageSize = (value as? Number)?.toInt()
                KEY_PATCH_SIZE -> patchSize = (value as? Number)?.toInt()
                KEY_MERGE_SIZE -> mergeSize = (value as? Number)?.toInt()
            }
            // Everything needed is present; skip the remaining keys.
            if (imageSize != null && patchSize != null && mergeSize != null) break
            i++
        }

        val image = imageSize
        val patch = patchSize
        if (image == null || patch == null || image <= 0 || patch <= 0) {
            Log.w(TAG, "mmproj declares no usable vision size (image=$imageSize patch=$patchSize)")
            return null
        }
        // spatial_merge_size is absent on towers that do not merge (e.g. Qwen2.5-VL).
        val merge = (mergeSize ?: 1).coerceAtLeast(1)
        if (image / patch / merge < 1) {
            Log.w(TAG, "mmproj vision geometry degenerate (image=$image patch=$patch merge=$merge)")
            return null
        }
        return GgufVisionConfig(imageSize = image, patchSize = patch, spatialMergeSize = merge)
    }

    /** Returns the parsed value, or null on an unknown type tag (which makes the rest unparseable). */
    private fun readValue(
        raf: RandomAccessFile,
        type: Int,
    ): Any? =
        when (type) {
            TYPE_UINT8, TYPE_INT8 -> {
                raf.readByte().toInt()
            }

            TYPE_UINT16, TYPE_INT16 -> {
                readLE(raf, 2).short.toInt()
            }

            TYPE_UINT32, TYPE_INT32 -> {
                readU32(raf)
            }

            TYPE_FLOAT32 -> {
                readLE(raf, 4).float
            }

            TYPE_BOOL -> {
                raf.readByte().toInt() != 0
            }

            TYPE_UINT64, TYPE_INT64 -> {
                readU64(raf)
            }

            TYPE_FLOAT64 -> {
                readLE(raf, 8).double
            }

            TYPE_STRING -> {
                readString(raf)
            }

            TYPE_ARRAY -> {
                skipArray(raf)
            }

            else -> {
                Log.w(TAG, "unknown GGUF value type $type")
                null
            }
        }

    /**
     * Skips an array value wholesale. None of the keys of interest are arrays, and
     * the contents are irrelevant — but the bytes still have to be consumed so the
     * following key parses. Returns a placeholder on success, null on a bad tag.
     */
    private fun skipArray(raf: RandomAccessFile): Any? {
        val elemType = readU32(raf).toInt()
        val count = readU64(raf)
        if (elemType == TYPE_STRING) {
            for (i in 0 until count) {
                if (readString(raf) == null) return null
            }
        } else {
            val width =
                when (elemType) {
                    TYPE_UINT8, TYPE_INT8, TYPE_BOOL -> {
                        1L
                    }

                    TYPE_UINT16, TYPE_INT16 -> {
                        2L
                    }

                    TYPE_UINT32, TYPE_INT32, TYPE_FLOAT32 -> {
                        4L
                    }

                    TYPE_UINT64, TYPE_INT64, TYPE_FLOAT64 -> {
                        8L
                    }

                    else -> {
                        Log.w(TAG, "unknown GGUF array element type $elemType")
                        return null
                    }
                }
            raf.seek(raf.filePointer + width * count)
        }
        return Unit
    }

    private fun readString(raf: RandomAccessFile): String? {
        val len = readU64(raf)
        // A sane key/value string; anything larger means the stream is desynced.
        if (len < 0 || len > 1L shl 20) {
            Log.w(TAG, "implausible GGUF string length $len")
            return null
        }
        val bytes = ByteArray(len.toInt())
        raf.readFully(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    private fun readU32(raf: RandomAccessFile): Long = readLE(raf, 4).int.toLong() and 0xFFFFFFFFL

    private fun readU64(raf: RandomAccessFile): Long = readLE(raf, 8).long

    private fun readLE(
        raf: RandomAccessFile,
        n: Int,
    ): ByteBuffer {
        val bytes = ByteArray(n)
        raf.readFully(bytes)
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    }
}
