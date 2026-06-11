[![Qualcomm® AI Hub Apps](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

# GenieX Demo

On-device AI chat application for Android on Snapdragon® powered by the
[GenieX SDK](https://github.com/qualcomm/geniex). Runs Large Language Models
(LLMs) and Vision-Language Models (VLMs) on the Snapdragon® Neural Processing
Unit (NPU), GPU, or CPU through a single pluggable runtime.

## Current limitations

> [!IMPORTANT]
> This app requires a **mobile** device with Snapdragon® 8 Elite or
> Snapdragon® 8 Elite Gen 5 for NPU acceleration. Other Snapdragon® chipsets
> fall back to CPU/GPU execution. Minimum Android API level is set by the
> shared `apps/_shared/scripts/versions.env` (`ANDROID_MIN_API`).

We recommend using a device from [QDC](https://qdc.qualcomm.com/) for this
demo.

## Prerequisites

1. Clone this repository.
2. Download [Android Studio](https://developer.android.com/studio).
   **Version 2024.3.1 or newer** is required.
3. [Enable USB debugging](https://developer.android.com/studio/debug/dev-options) on your Android device.

The GenieX Android binding is consumed from Maven Central as
`com.qualcomm.qti:geniex-android` (version pinned in
[`build.gradle`](build.gradle)). Gradle resolves it automatically on first
sync — no separate SDK download is required.

## Build the APK

1. Open this folder in Android Studio.
2. Run Gradle sync.
3. Build the `app` target (`Build` → `Build Bundle(s) / APK(s)` → `Build APK(s)`).
4. The APK is written to:

   ```text
   apps/geniex_chat_android/build/outputs/apk/{build_type}/app-{build_type}.apk
   ```

## Run on Device

1. Install the APK via `adb`:

   ```bash
   adb install -t apps/geniex_chat_android/build/outputs/apk/debug/app-debug.apk
   ```

2. Launch the app from the device's launcher. Models are downloaded on first
   use; pick one from the in-app catalog.

## Supported Hardware

- **NPU**: Qualcomm® Snapdragon® 8 Elite, Snapdragon® 8 Elite Gen 5
- **GPU**: Qualcomm® Adreno™ GPU
- **CPU**: ARM64-v8a

## Technologies Used by this App

- [Android SDK](https://developer.android.com/studio)
- [GenieX SDK](https://github.com/qualcomm/geniex) (Maven Central:
  `com.qualcomm.qti:geniex-android`)

## License

This app is released under the [BSD-3 License](../../../LICENSE) found at the
root of this repository.

The GenieX SDK dependency is released under its own license. Refer to the
[GenieX repository](https://github.com/qualcomm/geniex) for details.
