// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.objectdetection;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.DialogFragment;
import androidx.fragment.app.Fragment;

import org.opencv.android.OpenCVLoader;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

public class CameraFragment extends Fragment {
    private long lastTic = 0;

    private FragmentRender mFragmentRender;
    private float fps = 0;
    private String mCameraId;
    private static final String FRAGMENT_DIALOG = "dialog";
    private ObjectDetection detector;
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    private TextureView mTextureView;

    /**
     * {@link TextureView.SurfaceTextureListener} handles several lifecycle events on a {@link
     * TextureView}.
     */
    private final TextureView.SurfaceTextureListener mSurfaceTextureListener =
            new TextureView.SurfaceTextureListener() {

                @Override
                public void onSurfaceTextureAvailable(
                        @NonNull SurfaceTexture texture, int width, int height) {
                    openCamera();
                }

                @Override
                public void onSurfaceTextureSizeChanged(
                        @NonNull SurfaceTexture texture, int width, int height) {}

                @Override
                public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture texture) {
                    return true;
                }

                @Override
                public void onSurfaceTextureUpdated(@NonNull SurfaceTexture texture) {}
            };

    /** A {@link CameraCaptureSession } for camera preview. */
    private CameraCaptureSession mCaptureSession;

    /** A reference to the opened {@link CameraDevice}. */
    private CameraDevice mCameraDevice;

    /** The {@link android.util.Size} of camera preview. */
    private Size mPreviewSize;

    /** {@link CameraDevice.StateCallback} is called when {@link CameraDevice} changes its state. */
    private final CameraDevice.StateCallback mStateCallback =
            new CameraDevice.StateCallback() {

                @Override
                public void onOpened(@NonNull CameraDevice cameraDevice) {
                    // This method is called when the camera is opened.  We start camera preview
                    // here.
                    mCameraOpenCloseLock.release();
                    mCameraDevice = cameraDevice;
                    createCameraPreviewSession();
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice cameraDevice) {
                    mCameraOpenCloseLock.release();
                    cameraDevice.close();
                    mCameraDevice = null;
                }

                @Override
                public void onError(@NonNull CameraDevice cameraDevice, int error) {
                    mCameraOpenCloseLock.release();
                    cameraDevice.close();
                    mCameraDevice = null;
                    Activity activity = getActivity();
                    if (null != activity) {
                        activity.finish();
                    }
                }
            };

    public static CameraFragment create(ObjectDetection detector) {
        final CameraFragment fragment = new CameraFragment();
        fragment.detector = detector;
        return fragment;
    }

    /** An additional thread for running tasks that shouldn't block the UI. */
    private HandlerThread mBackgroundThread;

    /** A {@link Handler} for running tasks in the background. */
    private Handler mBackgroundHandler;

    /** {@link CaptureRequest.Builder} for the camera preview */
    private CaptureRequest.Builder mPreviewRequestBuilder;

    /** A {@link Semaphore} to prevent the app from exiting before closing the camera. */
    private final Semaphore mCameraOpenCloseLock = new Semaphore(1);

    /** Orientation of the camera sensor in degrees */
    private int mSensorOrientation;

    /** Orientation of the device (1= TODO */
    private int mDeviceOrientation;

    private int mFinalRotation;

    /**
     * Given {@code choices} of {@code Size}s supported by a camera, choose the smallest one that is
     * at least as large as the minimum width/height.
     *
     * @param choices The list of sizes that the camera supports for the intended output class
     * @param minWidth Minimum width
     * @param minHeight Minimum height
     * @return The smallest {@code Size} satisfying the minimum constraints
     */
    private static Size chooseOptimalSize(Size[] choices, int minWidth, int minHeight) {

        // Collect the supported resolutions that are at least as the target size
        List<Size> bigEnough = new ArrayList<>();
        for (Size option : choices) {
            if (option.getWidth() >= minWidth && option.getHeight() >= minHeight) {
                bigEnough.add(option);
            }
        }

        // Pick the smallest of those big enough. If there is no one big enough, we error out
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else {
            throw new RuntimeException("No suitable camera size found");
        }
    }

    private ActivityResultLauncher<String> requestPermissionLauncher;

    @Override
    public View onCreateView(
            LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        OpenCVLoader.initDebug();

        requestPermissionLauncher =
                registerForActivityResult(
                        new ActivityResultContracts.RequestPermission(),
                        isGranted -> {
                            if (!isGranted) {
                                // Permission has been denied, show an error dialog or a message
                                ErrorDialog.newInstance(getString(R.string.request_permission))
                                        .show(getChildFragmentManager(), FRAGMENT_DIALOG);
                            }
                        });
        return inflater.inflate(R.layout.fragment_camera, container, false);
    }

    @Override
    public void onViewCreated(@NonNull final View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        mTextureView = view.findViewById(R.id.surface);
        mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        mFragmentRender = view.findViewById(R.id.fragmentRender);
    }

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();

        // When the screen is turned off and turned back on, the SurfaceTexture is already
        // available, and "onSurfaceTextureAvailable" will not be called. In that case, we can open
        // a camera and start preview from here (otherwise, we wait until the surface is ready in
        // the SurfaceTextureListener).
        if (mTextureView.isAvailable()) {
            openCamera();
        } else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    @Override
    public void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    @Override
    public void onDestroy() {
        stopBackgroundThread();
        closeCamera();
        super.onDestroy();
    }

    private void requestCameraPermission() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            new ConfirmationDialog().show(getChildFragmentManager(), FRAGMENT_DIALOG);
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    /** Sets up member variables related to camera. */
    private void setUpCameraOutputs() {
        Activity activity = getActivity();
        assert activity != null;
        CameraManager mCameraManager =
                (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            for (String cameraId : mCameraManager.getCameraIdList()) {
                CameraCharacteristics characteristics =
                        mCameraManager.getCameraCharacteristics(cameraId);

                // We don't use a front facing camera in this sample.
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                StreamConfigurationMap map =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (map == null) {
                    continue;
                }

                // Get sensor orientation. Depending on the sensor orientation, we may need to
                // rotate the input image. This happens in ObjectDetector
                Integer sensorOrientation =
                        characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
                if (sensorOrientation != null) {
                    mSensorOrientation = sensorOrientation;
                }

                Size[] surfaceSizes = map.getOutputSizes(SurfaceTexture.class);

                assert (detector != null);
                int targetWidth, targetHeight;
                if (mSensorOrientation == 90 || mSensorOrientation == 270) {
                    targetWidth = detector.getInputHeight();
                    targetHeight = detector.getInputWidth();
                } else {
                    targetWidth = detector.getInputWidth();
                    targetHeight = detector.getInputHeight();
                }

                // Select camera feed image size to be close to the target size
                mPreviewSize = chooseOptimalSize(surfaceSizes, targetWidth, targetHeight);
                mCameraId = cameraId;
                return;
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        } catch (NullPointerException e) {
            // Currently an NPE is thrown when the Camera2API is used but not supported on the
            // device this code runs.
            ErrorDialog.newInstance(getString(R.string.camera_error))
                    .show(getChildFragmentManager(), FRAGMENT_DIALOG);
        }
    }

    /** Opens the camera specified by {@link CameraFragment#mCameraId}. */
    private void openCamera() {
        if (ContextCompat.checkSelfPermission(requireActivity(), Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
            return;
        }

        setUpCameraOutputs();
        Activity activity = requireActivity();
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            manager.openCamera(mCameraId, mStateCallback, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.", e);
        }
    }

    /** Closes the current {@link CameraDevice}. */
    private void closeCamera() {
        try {
            mCameraOpenCloseLock.acquire();
            if (null != mCaptureSession) {
                mCaptureSession.close();
                mCaptureSession = null;
            }
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            mCameraOpenCloseLock.release();
        }
    }

    /** Starts a background thread and its {@link Handler}. */
    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    /** Stops the background thread and its {@link Handler}. */
    private void stopBackgroundThread() {

        if (mBackgroundThread != null) {
            mBackgroundThread.quitSafely();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        }
    }

    /** Creates a new {@link CameraCaptureSession} for camera preview. */
    private void createCameraPreviewSession() {
        try {
            float viewWidth = mTextureView.getWidth();
            float viewHeight = mTextureView.getHeight();

            float previewWidth = mPreviewSize.getWidth();
            float previewHeight = mPreviewSize.getHeight();

            float scaleX = (float) viewWidth / previewWidth;
            float scaleY = (float) viewHeight / previewHeight;
            float scale = Math.max(scaleX, scaleY);

            float scaledWidth = scaleX * previewWidth;
            float scaledHeight = scaleY * previewWidth;

            float dx = (viewWidth - scaledWidth) / 2;
            float dy = (viewHeight - scaledHeight) / 2;

            Matrix matrix = new Matrix();
            matrix.setScale(scaleX * 0.25f, scaleY);
            matrix.postTranslate(dx, dy);

            mTextureView.setTransform(matrix);

            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;

            // We configure the size of default buffer to be the size of camera preview we want.
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());

            // This is the output Surface we need to start preview.
            Surface surface = new Surface(texture);

            // We set up a CaptureRequest.Builder with the output Surface.
            mPreviewRequestBuilder =
                    mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mPreviewRequestBuilder.addTarget(surface);

            try {
                mCameraDevice.createCaptureSession(List.of(surface), new CameraCapture(), null);
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    /** Compares two {@code Size}s based on their areas. */
    static class CompareSizesByArea implements Comparator<Size> {

        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum(
                    (long) lhs.getWidth() * lhs.getHeight()
                            - (long) rhs.getWidth() * rhs.getHeight());
        }
    }

    /** Shows an error message dialog. */
    public static class ErrorDialog extends DialogFragment {

        private static final String ARG_MESSAGE = "message";

        public static ErrorDialog newInstance(String message) {
            ErrorDialog dialog = new ErrorDialog();
            Bundle args = new Bundle();
            args.putString(ARG_MESSAGE, message);
            dialog.setArguments(args);
            return dialog;
        }

        @NonNull
        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Activity activity = getActivity();
            return new AlertDialog.Builder(activity)
                    .setMessage(requireArguments().getString(ARG_MESSAGE))
                    .setPositiveButton(
                            android.R.string.ok,
                            (dialogInterface, i) -> Objects.requireNonNull(activity).finish())
                    .create();
        }
    }

    /** Shows OK/Cancel confirmation dialog about camera permission. */
    public static class ConfirmationDialog extends DialogFragment {

        private ActivityResultLauncher<String> requestPermissionLauncher;

        @Override
        public void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);

            // Initialize the ActivityResultLauncher
            requestPermissionLauncher =
                    registerForActivityResult(
                            new ActivityResultContracts.RequestPermission(),
                            isGranted -> {
                                if (!isGranted) {
                                    // Permission denied, close the parent activity if it's not null
                                    Activity activity = getActivity();
                                    if (activity != null) {
                                        activity.finish();
                                    }
                                }
                            });
        }

        @NonNull
        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Fragment parent = getParentFragment();
            return new AlertDialog.Builder(getActivity())
                    .setMessage(R.string.request_permission)
                    .setPositiveButton(
                            android.R.string.ok,
                            (dialog, which) ->
                                    requestPermissionLauncher.launch(Manifest.permission.CAMERA))
                    .setNegativeButton(
                            android.R.string.cancel,
                            (dialog, which) -> {
                                Activity activity = Objects.requireNonNull(parent).getActivity();
                                if (activity != null) {
                                    activity.finish();
                                }
                            })
                    .create();
        }
    }

    public class CameraCapture extends android.hardware.camera2.CameraCaptureSession.StateCallback {
        @Override
        public void onConfigured(
                @NonNull android.hardware.camera2.CameraCaptureSession cameraCaptureSession) {
            mCaptureSession = cameraCaptureSession;

            try {
                // Auto focus should be continuous for camera preview.
                mPreviewRequestBuilder.set(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                // Finally, we start displaying the camera preview.
                cameraCaptureSession.setRepeatingRequest(
                        mPreviewRequestBuilder.build(), new CameraSession(), mBackgroundHandler);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        @Override
        public void onConfigureFailed(
                @NonNull android.hardware.camera2.CameraCaptureSession cameraCaptureSession) {}
    }

    /**
     * Retrieves the JPEG orientation from the specified screen rotation.
     *
     * @param rotation The screen rotation.
     * @return The JPEG orientation (one of 0, 90, 270, and 360)
     */
    private int getOrientation(int rotation) {
        // Sensor orientation is 90 for most devices, or 270 for some devices (eg. Nexus 5X)
        // We have to take that into account and rotate JPEG properly.
        // For devices with orientation of 90, we simply return our mapping from ORIENTATIONS.
        // For devices with orientation of 270, we need to rotate the JPEG 180 degrees.
        return (ORIENTATIONS.get(rotation) + mSensorOrientation + 270) % 360;
    }

    private class CameraSession
            extends android.hardware.camera2.CameraCaptureSession.CaptureCallback {

        @Override
        public void onCaptureCompleted(
                @NonNull CameraCaptureSession session,
                @NonNull CaptureRequest request,
                @NonNull TotalCaptureResult result) {

            super.onCaptureCompleted(session, request, result);

            if (lastTic == 0) {
                lastTic = System.currentTimeMillis();
            } else {
                long newTic = System.currentTimeMillis();
                if (lastTic != newTic) {
                    fps = 1000.f / (float) (newTic - lastTic);
                }
                lastTic = newTic;
            }

            if (detector != null) {
                // Save camera feed to Bitmap at its native resolution
                Bitmap mBitmap = mTextureView.getBitmap();
                if (mBitmap == null) {
                    return;
                }

                ArrayList<RectangleBox> BBlist = new ArrayList<>();

                mDeviceOrientation = getResources().getConfiguration().orientation;
                final Activity activity = getActivity();
                int displayRotation = activity.getWindowManager().getDefaultDisplay().getRotation();

                int orient = getOrientation(displayRotation);
                mFinalRotation = orient;

                detector.predict(mBitmap, orient, BBlist);
                mFragmentRender.setCoordsList(BBlist);
                mFragmentRender.render(
                        mBitmap,
                        mPreviewSize,
                        fps,
                        detector.getLastInferenceTime(),
                        detector.getLastPreprocessingTime(),
                        detector.getLastPostprocessingTime(),
                        displayRotation);
            }
        }

        @Override
        public void onCaptureFailed(
                @NonNull CameraCaptureSession session,
                @NonNull CaptureRequest request,
                @NonNull CaptureFailure failure) {
            super.onCaptureFailed(session, request, failure);
        }

        @Override
        public void onCaptureProgressed(
                @NonNull CameraCaptureSession session,
                @NonNull CaptureRequest request,
                @NonNull CaptureResult partialResult) {
            super.onCaptureProgressed(session, request, partialResult);
        }

        @Override
        public void onCaptureStarted(
                @NonNull CameraCaptureSession session,
                @NonNull CaptureRequest request,
                long timestamp,
                long frameNumber) {
            super.onCaptureStarted(session, request, timestamp, frameNumber);
        }
    }
}
